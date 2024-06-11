from functools import partial
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


class AtariFeatures(nn.Module):
    """ Configures the default Atari feature extractor. """

    def __init__(self, input_depth, hidden_size):
        super(AtariFeatures, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_depth, 32, kernel_size=8, stride=4),
            # nn.LayerNorm((32, 20, 20)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.LayerNorm((64, 9, 9)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.LayerNorm((64, 7, 7)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        # self.bn = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.convs(x).view(x.shape[0], -1)
        return F.relu(self.bn(self.fc(x)))


class InverseKinematicsNet(nn.Module):
    def __init__(
        self, action_no, input_ch=1, hist_len=4, hidden_size=256, resnet=False
    ):
        super(InverseKinematicsNet, self).__init__()
        Phi = AtariResNetFeatures if resnet else AtariFeatures
        self._phi = Phi(hist_len * input_ch, hidden_size)
        self._agg = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self._proj = nn.Sequential(
            nn.Linear(16 * 9 * 9, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self._classifier = nn.Linear(hidden_size, action_no)

    def forward(self, x):
        state, state_ = x
        N = state.shape[0]
        state = state.float().div(255)
        state_ = state_.float().div(255)

        # concatenate the two in a single batch for efficient inference
        x = torch.cat((state, state_))
        x = self._phi(x)  # (N*2, d)
        phi, phi_ = x.split(N, dim=0)
        # aggregate the resulting features
        # x = phi + phi_
        x = self._agg(torch.cat([phi, phi_], dim=1))
        x = self._proj(x.view(x.shape[0], -1))
        return self._classifier(x)


class InverseKinematicsDualNet(nn.Module):
    def __init__(
        self, action_no, input_ch=1, hist_len=4, hidden_size=256, resnet=False
    ):
        super(InverseKinematicsDualNet, self).__init__()
        Phi = AtariResNetFeatures if resnet else AtariFeatures
        self._phi = Phi(hist_len * input_ch, hidden_size)
        self._phi_ = Phi(hist_len * input_ch, hidden_size)
        # self._proj = nn.Linear(2 * hidden_size, hidden_size)
        self._agg = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self._proj = nn.Sequential(
            nn.Linear(16 * 9 * 9, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self._classifier = nn.Linear(hidden_size, action_no)

    def forward(self, x):
        state, state_ = x
        state = state.float().div(255)
        state_ = state_.float().div(255)

        phi = self._phi(state)
        phi_ = self._phi_(state_)
        x = self._agg(torch.cat([phi, phi_], dim=1))
        x = self._proj(x.view(x.shape[0], -1))
        # x = phi + phi_
        # x = self._relu(self._bn(self._proj(torch.cat([phi, phi_], dim=1))))
        return self._classifier(x)


class AtariResNetFeatures(nn.Module):
    """ Configures the default Atari feature extractor. """

    def __init__(self, input_depth, hidden_size):
        super(AtariResNetFeatures, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_depth, 32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            BasicBlock(
                32,
                64,
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(32, 64, 1, 2, bias=False), nn.BatchNorm2d(64)
                ),
            ),
            BasicBlock(
                64,
                64,
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(64, 64, 1, 2, bias=False), nn.BatchNorm2d(64)
                ),
            ),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
        )
        # self.fc = nn.Linear(16 * 9 * 9, hidden_size)
        # self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        return self.convs(x)
        # x = self.convs(x).view(x.shape[0], -1)
        # return F.relu(self.bn(self.fc(x)))


class AtariNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        action_no,
        input_ch=1,
        hist_len=4,
        hidden_size=256,
        initializer="xavier_uniform",
        support=None,
        **kwargs,
    ):
        super(AtariNet, self).__init__()

        assert initializer in (
            "xavier_uniform",
            "variance_scaling_uniform",
        ), "Only implements xavier_uniform and variance_scaling_uniform."

        self.__action_no = action_no
        self.__initializer = initializer
        self.__support = None
        if support is not None:
            self.__support = nn.Parameter(
                torch.linspace(*support), requires_grad=False
            )  # handy to make it a Parameter so that model.to(device) works
            out_size = action_no * len(self.__support)
        else:
            out_size = action_no

        # get the feature extractor and fully connected layers
        self.__features = get_feature_extractor(hist_len * input_ch, hidden_size)
        self.__head = nn.Linear(hidden_size, out_size)
        self.reset_parameters()

    def forward(self, x, probs=False, log_probs=False):
        # assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)
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


def variance_scaling_uniform_(tensor, scale=0.1, mode="fan_in"):
    # type: (Tensor, float) -> Tensor
    r"""Variance Scaling, as in Keras.

    Uniform sampling from `[-a, a]` where:

        `a = sqrt(3 * scale / n)`

    and `n` is the number of neurons according to the `mode`.

    """
    # pylint: disable=protected-access,invalid-name
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    a = 3 * scale
    a /= fan_in if mode == "fan_in" else fan_out
    weights = nn.init._no_grad_uniform_(tensor, -a, a)
    # pylint: enable=protected-access,invalid-name
    return weights
