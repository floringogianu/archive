""" Neural Network architecture for Atari games.
"""
from functools import partial, reduce
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = [
    "get_estimator",
    "MLP",
    "AtariNet",
    "MinAtarNet",
]


def get_feature_extractor(input_depth, activation_module):
    """ Configures the default Atari feature extractor. """
    convs = [
        nn.Conv2d(input_depth, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    ]

    return nn.Sequential(
        convs[0],
        activation_module(inplace=True),
        convs[1],
        activation_module(inplace=True),
        convs[2],
        activation_module(inplace=True),
    )


def get_head(hidden_size, out_size, activation_module, shared_bias=False):
    """ Configures the default Atari output layers. """
    fc0 = nn.Linear(64 * 7 * 7, hidden_size)
    fc1 = (
        SharedBiasLinear(hidden_size, out_size)
        if shared_bias
        else nn.Linear(hidden_size, out_size)
    )
    return nn.Sequential(fc0, activation_module(inplace=True), fc1)


def no_grad(module):
    """ Callback for turning off the gradient of a module.
    """
    try:
        module.weight.requires_grad = False
    except AttributeError:
        pass


def variance_scaling_uniform_(tensor, scale=0.1, mode="fan_in"):
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

    def __init__(self, action_no, layers=None, support=None, **kwargs):
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
        initializer="xavier_uniform",
        support=None,
        activation="ReLU",
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
        activation = getattr(nn, activation)
        self.__features = get_feature_extractor(hist_len * input_ch, activation)
        self.__head = get_head(hidden_size, out_size, activation, shared_bias)

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
            else partial(variance_scaling_uniform_, scale=1.0 / math.sqrt(3.0))
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


class MinAtarNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        action_no,
        input_ch=1,
        support=None,
        initializer="xavier_uniform",
        layer_dims=((16,), (128,)),
        activation="ReLU",
        **kwargs,
    ):
        super(MinAtarNet, self).__init__()

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

        # configure the net
        conv_layers, lin_layers = layer_dims
        feature_extractor, in_ch, out_wh = [], input_ch, 10
        activation = getattr(nn, activation)
        for out_ch in conv_layers:
            feature_extractor += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1),
                activation(inplace=True),
            ]
            in_ch = out_ch
            out_wh -= 2  # change this for a different kernel size or stride.
        self.__features = nn.Sequential(*feature_extractor)

        head, in_size = [], out_wh ** 2 * in_ch
        for hidden_size in lin_layers:
            head += [
                nn.Linear(in_size, hidden_size),
                activation(inplace=True),
            ]
            in_size = hidden_size
        head.append(nn.Linear(in_size, out_size))
        self.__head = nn.Sequential(*head)

        self.reset_parameters()

    def forward(self, x, probs=False, log_probs=False):
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"
        # assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float()
        if x.ndimension() == 5:
            x = x.squeeze(1)  # drop the "history"

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
            else partial(variance_scaling_uniform_, scale=1.0 / math.sqrt(3.0))
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


# Fancy Periodic Link Functions and Initialisations   ----------------------------------


@torch.jit.script
def periodic_relu(x):
    pi = torch.tensor(3.1415926535897932)
    return (
        8
        / pi ** 2
        * (
            ((x + pi / 2) - pi * torch.floor((x + pi / 2) / pi + 0.5))
            * (-1) ** torch.floor((x + pi / 2) / pi + 0.5)
            + (x - pi * torch.floor(x / pi + 0.5)) * (-1) ** torch.floor(x / pi + 0.5)
        )
    )


@torch.jit.script
def triangle(x):
    pi = torch.tensor(3.1415926535897932)
    pdiv2sqrt2 = torch.tensor(1.1107207345)
    return (
        pdiv2sqrt2
        * (x - pi * torch.floor(x / pi + 0.5))
        * (-1) ** torch.floor(x / pi + 0.5)
    )


@torch.jit.script
def invlink_uniform(x):
    pi = torch.tensor(3.1415926535897932)
    pi2 = torch.tensor(6.2831853071795864)
    return pi2 * torch.sigmoid(x) - pi


class PiReLU(nn.Module):
    r"""Applies the rectified linear unit function element-wise, periodically:
    """
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, **kwargs):
        super(PiReLU, self).__init__()

    def forward(self, input):
        return periodic_relu(input)


class Triangle(nn.Module):
    r"""Applies the rectified linear unit function element-wise, periodically:
    """
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, **kwargs):
        super(Triangle, self).__init__()

    def forward(self, input):
        return triangle(input)


def studentT_(w, nu=3 / 2):
    wdist = torch.distributions.StudentT(2 * nu)
    w.data = wdist.sample(w.shape)
    return w


PRIORS = {
    "RBF/PiReLU": (torch.nn.init.normal_, PiReLU),
    "Matern/PiReLU": (studentT_, PiReLU),
    "RBF/Triangle": (torch.nn.init.normal_, Triangle),
    "Matern/Triangle": (studentT_, Triangle),
    "PiReLU": (None, PiReLU),
    "Triangle": (None, Triangle),
}


class ConstrainedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super(ConstrainedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return F.linear(
            x,
            self.weight,
            invlink_uniform(self.bias) if self.bias is not None else self.bias,
        )

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class PeriodicMinAtarNet(nn.Module):
    def __init__(  # pylint: disable=bad-continuation
        self,
        action_no,
        prior="RBF/PiReLU",
        input_ch=1,
        conv_layers=(16,),
        bottleneck_width=64,
        model_layer_width=2048,
        activation="ReLU",
        **kwargs,
    ):
        super(PeriodicMinAtarNet, self).__init__()

        self.prior = prior
        self.model_layer_width = model_layer_width

        # configure the feature extractor
        feature_extractor, in_ch, out_wh = [], input_ch, 10
        activation = getattr(nn, activation)
        for out_ch in conv_layers:
            feature_extractor += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1),
                activation(inplace=True),
            ]
            in_ch = out_ch
            out_wh -= 2  # change this for a different kernel size or stride.
        feature_extractor.append(nn.Flatten())
        ft_size = out_wh ** 2 * in_ch

        # configure the bottleneck
        if bottleneck_width:
            feature_extractor += [
                nn.Linear(ft_size, bottleneck_width),
                activation(inplace=True),
            ]
            ft_size = bottleneck_width

        self._features = nn.Sequential(*feature_extractor)

        # configure the model layer
        _, link_fn = PRIORS.get(prior)
        self.model_layer = ConstrainedLinear(ft_size, model_layer_width, True)
        self.link_fn = link_fn(inplace=True)

        # output
        self.out = nn.Linear(model_layer_width, action_no)

        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        if x.ndimension() == 5:
            x = x.squeeze(1)  # drop the "history"
        x = self._features(x)
        # were the magic happens
        x = self.link_fn(self.model_layer(x))
        return self.out(x)

    def reset_parameters(self):
        for module in self._features.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()
        # model layer init
        weight_init_, _ = PRIORS.get(self.prior)
        if weight_init_ is not None:
            weight_init_(self.model_layer.weight)
            nn.init.uniform_(self.model_layer.bias, -math.pi, math.pi)
            # out layer init
            nn.init.normal_(
                self.out.weight, mean=0.0, std=1.0 / (self.model_layer_width ** 0.5)
            )
        else:
            nn.init.xavier_uniform_(self.model_layer.weight)
            nn.init.xavier_uniform_(self.out.weight)
            self.model_layer.bias.data.zero_()
            self.out.bias.data.zero_()

    def extra_repr(self) -> str:
        return "prior={}".format(self.prior)


FNAPPROX = {
    "MinAtarNet": MinAtarNet,
    "PeriodicMinAtarNet": PeriodicMinAtarNet,
    "AtariNet": AtariNet,
    "MLP": MLP,
}


def get_estimator(opt, env):
    """ Configure an estimator """
    estimator_args = opt.estimator.args
    if opt.estimator.name == "MLP":
        estimator_args["layers"] = [
            reduce(lambda x, y: x * y, env.observation_space.shape),
            *estimator_args["layers"],
        ]
    if opt.estimator.name in ("MinAtarNet", "PeriodicMinAtarNet"):
        estimator_args["input_ch"] = env.observation_space.shape[-1]
    estimator = FNAPPROX.get(opt.estimator.name)(opt.action_cnt, **estimator_args)
    estimator.to(opt.device)

    if (opt.agent.name == "DQN") and ("support" in opt.estimator.args):
        raise ValueError("DQN estimator should not have a support.")
    if (opt.agent.name == "C51") and ("support" not in opt.estimator.args):
        raise ValueError("C51 requires an estimator with support.")

    return estimator

