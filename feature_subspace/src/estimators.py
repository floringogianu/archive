""" Neural Network architecture for Atari games.
"""
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = [
    "MLP",
    "AtariNet",
    "MinAtarNet",
    "get_feature_extractor",
    "get_head",
]


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


class MLP(nn.Module):
    """ MLP estimator with variable depth and width for both C51 and DQN. """

    def __init__(  # pylint: disable=bad-continuation
        self, action_no, layers=None, support=None, **kwargs,
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
        support=None,
        **kwargs,
    ):
        super(AtariNet, self).__init__()

        self.__action_no = action_no
        self.__support = None
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
        init_ = nn.init.xavier_uniform_

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


class SubspaceConv(nn.Conv2d):
    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups,
        )
        return x


class TwoParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)


class LinesConv2D(TwoParamConv):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w


class SubspaceLinear(nn.Linear):
    def forward(self, x):
        w = self.get_weight()
        x = F.linear(x, w, self.bias)
        return x


class TwoParamLinear(SubspaceLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)


class LinesLinear(TwoParamLinear):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w


class MinAtarNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(
        self,
        action_no,
        input_ch=1,
        mode="point",
        support=None,
        layer_dims=((16,), (128,)),
        **kwargs,
    ):
        super(MinAtarNet, self).__init__()

        self.__action_no = action_no
        self.__mode = mode

        self.__support = None
        if support is not None:
            self.__support = nn.Parameter(
                torch.linspace(*support), requires_grad=False
            )  # handy to make it a Parameter so that model.to(device) works
            out_size = action_no * len(self.__support)
        else:
            out_size = action_no

        # configure the net
        Conv, Linear = (
            (nn.Conv2d, nn.Linear) if mode == "point" else (LinesConv2D, LinesLinear)
        )

        conv_layers, lin_layers = layer_dims
        feature_extractor, in_ch, out_wh = [], input_ch, 10
        for out_ch in conv_layers:
            feature_extractor += [
                Conv(in_ch, out_ch, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_wh -= 2  # change this for a different kernel size or stride.

        in_size = out_wh ** 2 * in_ch
        feature_extractor.append(nn.Flatten())
        for hidden_size in lin_layers:
            feature_extractor += [
                Linear(in_size, hidden_size),
                nn.ReLU(inplace=True),
            ]
            in_size = hidden_size

        self.__features = nn.Sequential(*feature_extractor)
        self.__head = nn.Linear(in_size, out_size)

        self.reset_parameters()

    def forward(self, x, probs=False, log_probs=False):
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"
        # assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float()
        if x.ndimension() == 5:
            x = x.squeeze(1)  # drop the "history"

        if self.__mode in ("line", "bezier"):
            alpha = torch.rand(1).item()
            for m in self.__features.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    setattr(m, "alpha", alpha)

        x = self.__features(x)

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
        init_ = nn.init.xavier_uniform_

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_(module.weight)
                module.bias.data.zero_()
                if hasattr(module, "initialize"):
                    module.initialize(init_)

    @property
    def feature_extractor(self):
        """ Return the feature extractor. """
        return self.__features

    @property
    def head(self):
        """ Return the head. """
        return self.__head
