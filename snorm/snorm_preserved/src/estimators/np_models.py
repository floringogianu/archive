""" Norm Preserving Estimators. """
from math import sqrt
from functools import partial
from itertools import chain

import torch
import torch.nn as nn

from .activations import MaxMin, GroupSort
from .snorm_models import hook_spectral_normalization, variance_scaling_uniform_
from .layers.bcop import BCOP
from .layers.bjork_linear import BjorckLinear

__all__ = ["NormPreservingMinAtarNet", "MinMaxAtariNet"]


class NormPreservingMinAtarNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(
        self,
        action_no,
        input_ch=1,
        hidden_size=128,
        bjorck_iters=5,
        support=None,
        **kwargs,
    ):
        super().__init__()

        self.__scaling = 1.0 ** (1 / 4)
        self.__action_no = action_no
        self.__support = None

        if support is not None:
            self.__support = nn.Parameter(
                torch.linspace(*support), requires_grad=False
            )  # handy to make it a Parameter so that model.to(device) works
            out_size = action_no * len(self.__support)
        else:
            out_size = action_no

        self.conv1 = BCOP(
            input_ch,
            out_channels=16,
            kernel_size=4,
            stride=1,
            padding=2,
            bjorck_iters=bjorck_iters,
        )
        self.activation = GroupSort(group_size=2, axis=1, new_impl=True)
        self.fc1 = BjorckLinear(
            10 * 10 * 16, hidden_size, bjorck_iters=bjorck_iters + 2
        )
        self.out = nn.Linear(hidden_size, out_size)
        self.spectral = None

    def forward(self, x, probs=False, log_probs=False):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float()
        if x.ndimension() == 5:
            x = x.squeeze(1)  # drop the "history"

        x = self.conv1(x)
        x = x * self.__scaling
        x = self.activation(x)
        x = self.fc1(x.flatten(start_dim=1))
        x = x * self.__scaling
        x = self.activation(x)
        qs = self.out(x)

        # distributional RL
        # either return p(s,·), log(p(s,·)) or the distributional Q(s,·)
        if self.__support is not None:
            logits = qs.view(qs.shape[0], self.__action_no, len(self.support))
            if probs:
                return torch.softmax(logits, dim=2)
            if log_probs:
                return torch.log_softmax(logits, dim=2)
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # or just return Q(s,a)
        return qs

    @property
    def support(self):
        """ Return the support of the Q-Value distribution. """
        return self.__support


class MinMaxAtariNet(nn.Module):
    def __init__(  # pylint: disable=bad-continuation
        self,
        action_no,
        input_ch=1,
        hist_len=4,
        hidden_size=256,
        support=None,
        spectral=None,
        **kwargs,
    ):
        super().__init__()

        self.__action_no = action_no
        self.__support = None
        self.spectral = spectral

        self.conv1 = BCOP(input_ch, out_channels=16, kernel_size=4, stride=2)

        if support is not None:
            self.__support = nn.Parameter(
                torch.linspace(*support), requires_grad=False
            )  # handy to make it a Parameter so that model.to(device) works
            out_size = action_no * len(self.__support)
        else:
            out_size = action_no

        self.__features = nn.Sequential(
            nn.Conv2d(input_ch * hist_len, 32, kernel_size=8, stride=4),
            MaxMin(16, axis=1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            MaxMin(32, axis=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            MaxMin(32, axis=1),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_size),
            MaxMin(hidden_size // 2),
            nn.Linear(hidden_size, out_size),
        )

        self.reset_parameters()

        # We allways compute spectral norm except when None or notrace
        if spectral is not None:
            self.__hooked_layers = hook_spectral_normalization(
                spectral,
                chain(self.__features.named_children(), self.__head.named_children()),
                **kwargs,
            )

    def forward(self, x, probs=False, log_probs=False):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div_(255)
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"

        x = self.__features(x)
        x = x.view(x.size(0), -1)
        qs = self.__head(x)

        # distributional RL
        # either return p(s,·), log(p(s,·)) or the distributional Q(s,·)
        if self.__support is not None:
            logits = qs.view(qs.shape[0], self.__action_no, len(self.support))
            if probs:
                return torch.softmax(logits, dim=2)
            if log_probs:
                return torch.log_softmax(logits, dim=2)
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # or just return Q(s,a)
        return qs

    @property
    def support(self):
        """ Return the support of the Q-Value distribution. """
        return self.__support

    def _hook_spectral_normalization(self, spectral):
        """ Hooks spectral norm, even if just for logging the value.
            For certain values of spectral it will also acutally apply the
            weight normalization (the `active_layers` below).
        """

    def get_spectral_norms(self):
        """ Return the spectral norms of layers hooked on spectral norm. """
        return {
            str(idx): layer.weight_sigma.item() for idx, layer in self.__hooked_layers
        }

    def reset_parameters(self):
        """ Weight init.
        """
        init_ = (
            nn.init.xavier_uniform_
            if self.__initializer == "xavier_uniform"
            else partial(variance_scaling_uniform_, scale=1.0 / sqrt(3.0))
        )

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_(module.weight)
                module.bias.data.zero_()

    @property
    def feature_extractor(self):
        """ Return the feature extractor. """
        return self.__features

    @property
    def head(self):
        """ Return the layers used as heads in Bootstrapped DQN. """
        return self.__head
