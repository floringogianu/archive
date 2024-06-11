""" Neural Network architecture for Atari games.
"""
from collections import OrderedDict
from functools import partial
from itertools import chain
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.conv_spectral_norm import spectral_norm_conv2d
from src.linear_spectral_norm import spectral_norm

__all__ = [
    "MLP",
    "AtariNet",
    "get_feature_extractor",
    "get_head",
]


def hook_spectral_normalization(  # pylint: disable=bad-continuation
    spectral, layers, leave_smaller=False, lipschitz_k=1, flow_through_norm="linear"
):
    """Uses the convention in `spectral` to hook spectral normalization on
    modules in `layers`.

    Args:
        spectral (str): A string of negative indices. Ex.: `-1` or `-2,-3`.
            To hook spectral normalization only for computing the norm
            and not applying it on the weights add the identifier `L`.
            Ex.: `-1L`, `-2,-3,-4L`.
        layers (list): Ordered list of tuples of (module_name, nn.Module).

    Returns:
        normalized: Layers
    """
    # Filter unsupported layers
    layers = [
        (n, m)
        for (n, m) in layers
        if isinstance(m, (nn.Conv2d, nn.Linear, SharedBiasLinear))
    ]
    N = len(layers)

    # Some convenient conventions
    if spectral == "":
        # log all layers, but do not apply
        spectral = ",".join([f"-{i}L" for i in range(N)])
    elif spectral == "full":
        # apply snorm everywhere
        spectral = ",".join([f"-{i}" for i in range(N)])
    else:
        spectral = str(spectral)  # sometimes this is just a number eg.: -3

    # [('-2', True), ('-3L', False)]
    layers_status = [(i, "L" not in i) for i in spectral.split(",")]
    # For len(layers)=5: [(3, True), (4, False)]
    layers_status = [(int(i if s else i[:-1]) % N, s) for i, s in layers_status]

    hooked_layers = []
    conv_flow_through_norm = flow_through_norm in [True, "conv", "all"]
    linear_flow_through_norm = flow_through_norm in [True, "linear", "all"]

    for (idx, active) in layers_status:
        layer_name, layer = layers[idx]

        if isinstance(layer, nn.Conv2d):
            spectral_norm_conv2d(
                layer,
                active=active,
                leave_smaller=leave_smaller,
                lipschitz_k=lipschitz_k,
                flow_through_norm=conv_flow_through_norm,
            )
        elif isinstance(layer, (nn.Linear, SharedBiasLinear)):
            spectral_norm(
                layer,
                active=active,
                leave_smaller=leave_smaller,
                lipschitz_k=lipschitz_k,
                flow_through_norm=linear_flow_through_norm,
            )
        else:
            raise NotImplementedError(
                "S-Norm on {} layer type not implemented for {} @ ({}): {}".format(
                    type(layer), idx, layer_name, layer
                )
            )
        hooked_layers.append((idx, layer))

        print(
            "{} SNorm to {} @ ({}): {}".format(
                "Active " if active else "Logging", idx, layer_name, layer
            )
        )
    return hooked_layers


def get_feature_extractor(input_depth):
    """ Configures the default Atari feature extractor. """
    convs = [
        nn.Conv2d(input_depth, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    ]

    return nn.Sequential(
        convs[0],
        nn.ReLU(inplace=True),
        convs[1],
        nn.ReLU(inplace=True),
        convs[2],
        nn.ReLU(inplace=True),
    )


def get_head(hidden_size, out_size, shared_bias=False):
    """ Configures the default Atari output layers. """
    fc0 = nn.Linear(64 * 7 * 7, hidden_size)
    fc1 = (
        SharedBiasLinear(hidden_size, out_size)
        if shared_bias
        else nn.Linear(hidden_size, out_size)
    )
    return nn.Sequential(fc0, nn.ReLU(inplace=True), fc1)


def no_grad(module):
    """ Callback for turning off the gradient of a module.
    """
    try:
        module.weight.requires_grad = False
    except AttributeError:
        pass


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


class MLP(nn.Module):
    """ MLP estimator with variable depth and width for both C51 and DQN. """

    def __init__(  # pylint: disable=bad-continuation
        self, action_no, layers=None, support=None, spectral=None, **kwargs,
    ):
        super(MLP, self).__init__()

        self.action_no = action_no
        dims = [*layers, action_no]
        # create support and adjust dims in case of distributional
        if support is not None:
            # handy to make it a Parameter so that model.to(device) works
            self.support = nn.Parameter(torch.linspace(*support), requires_grad=False)
            dims[-1] = self.action_no * len(self.support)
        else:
            self.support = None
        # create MLP layers
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims))]
        )

        if spectral is not None:
            self.hooked_layers = hook_spectral_normalization(
                spectral, self.layers.named_children(), **kwargs,
            )

    def forward(self, x, probs=False, log_probs=False):
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"

        x = x.view(x.shape[0], -1)  # usually it comes with a history dimension

        for module in self.layers[:-1]:
            x = F.relu(module(x), inplace=True)
        qs = self.layers[-1](x)

        # distributional RL
        # either return p(s,·), log(p(s,·)) or the distributional Q(s,·)
        if self.support is not None:
            logits = qs.view(qs.shape[0], self.action_no, len(self.support))
            if probs:
                return torch.softmax(logits, dim=2)
            if log_probs:
                return torch.log_softmax(logits, dim=2)
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # or just return Q(s,a)
        return qs

    def get_spectral_norms(self):
        """ Return the spectral norms of layers hooked on spectral norm. """
        return {
            str(idx): layer.weight_sigma.item() for idx, layer in self.hooked_layers
        }


class SharedBiasLinear(nn.Linear):
    """ Applies a linear transformation to the incoming data: `y = xA^T + b`.
        As opposed to the default Linear layer it has a shared bias term.
        This is employed for example in Double-DQN.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features, out_features):
        super(SharedBiasLinear, self).__init__(in_features, out_features, True)
        self.bias = Parameter(torch.Tensor(1))

    def extra_repr(self):
        return "in_features={}, out_features={}, bias=shared".format(
            self.in_features, self.out_features
        )


class AtariNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        action_no,
        input_ch=1,
        hist_len=4,
        hidden_size=256,
        shared_bias=False,
        initializer="xavier_uniform",
        support=None,
        spectral=None,
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
        self.spectral = spectral
        if support is not None:
            self.__support = nn.Parameter(
                torch.linspace(*support), requires_grad=False
            )  # handy to make it a Parameter so that model.to(device) works
            out_size = action_no * len(self.__support)
        else:
            out_size = action_no

        # get the feature extractor and fully connected layers
        self.__features = get_feature_extractor(hist_len * input_ch)
        self.__head = get_head(hidden_size, out_size, shared_bias)

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


class SVHNMlp(nn.Module):
    """ yada, yada, mlp.
    """

    def __init__(self, in_shape=(3, 32, 32), hidden=(128,), nclasses=10, spectral=None):
        super().__init__()

        nin = in_shape[0] * in_shape[1] * in_shape[2]
        layers = []
        for nout in hidden:
            layers.append(nn.Linear(nin, nout))
            layers.append(nn.ReLU())
            nin = nout
        layers.append(nn.Linear(nin, nclasses))
        self._layers = nn.Sequential(*layers)

        # We allways compute spectral norm except when None or notrace
        if spectral is not None:
            self.__hooked_layers = hook_spectral_normalization(
                spectral, self._layers.named_children()
            )
        else:
            self.__hooked_layers = []

    def forward(self, x):
        return self._layers(torch.flatten(x, start_dim=1))

    def get_spectral_norms(self):
        """ Return the spectral norms of layers hooked on spectral norm.
        """
        return OrderedDict(
            [
                (str(idx), layer.weight_sigma.item())
                for idx, layer in self.__hooked_layers
            ]
        )


class SVHNNet(nn.Module):
    """ yada, yada, convnet.
    """

    def __init__(
        self,
        in_shape=(3, 32, 32),
        conv_scale=1,
        deep_conv=True,
        hidden=(128,),
        nclasses=10,
        spectral=None,
    ):
        super().__init__()
        if deep_conv:
            self._features = nn.Sequential(
                nn.Conv2d(3, 16 * conv_scale, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16 * conv_scale, 32 * conv_scale, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32 * conv_scale, 64 * conv_scale, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            nin = 256 * conv_scale
        else:
            self._features = nn.Sequential(
                nn.Conv2d(3, 16 * conv_scale, 5),  # 16x28x26
                nn.ReLU(),
                nn.MaxPool2d(2),  # 16x14x14
                nn.Conv2d(16 * conv_scale, 32 * conv_scale, 5),  # 32x10x10
                nn.ReLU(),
                nn.MaxPool2d(2),  # 32x5x5
                nn.Conv2d(32 * conv_scale, 64 * conv_scale, 3),  # 64x3x3
                nn.ReLU(),
            )
            nin = 576 * conv_scale
        head_layers = []
        for nout in hidden:
            head_layers.append(nn.Linear(nin, nout))
            head_layers.append(nn.ReLU())
            nin = nout
        head_layers.append(nn.Linear(nin, nclasses))
        self._head = nn.Sequential(*head_layers)

        # We allways compute spectral norm except when None or notrace
        if spectral is not None:
            self.__hooked_layers = hook_spectral_normalization(
                spectral,
                chain(self._features.named_children(), self._head.named_children()),
            )
        else:
            self.__hooked_layers = []

    def forward(self, x):
        x = self._features(x)
        x = torch.flatten(x, start_dim=1)
        return self._head(x)

    def get_spectral_norms(self):
        """ Return the spectral norms of layers hooked on spectral norm.
        """
        return OrderedDict(
            [
                (str(idx), layer.weight_sigma.item())
                for idx, layer in self.__hooked_layers
            ]
        )
