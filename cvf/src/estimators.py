""" Neural Network architecture for Atari games.
"""
import torch
import torch.nn as nn
from gym.envs.registration import spec
from torch.nn.parameter import Parameter

__all__ = [
    "AtariNet",
    "get_feature_extractor",
    "get_head",
]


def get_feature_extractor(input_depth, spectral_norm=False):
    """ Configures the default Atari feature extractor. """
    convs = [
        nn.Conv2d(input_depth, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    ]
    if spectral_norm:
        convs = [nn.utils.spectral_norm(c) for c in convs]

    return nn.Sequential(
        convs[0],
        nn.ReLU(inplace=True),
        convs[1],
        nn.ReLU(inplace=True),
        convs[2],
        nn.ReLU(inplace=True),
    )


def get_head(hidden_size, out_size, shared_bias=False, spectral_norm=False):
    """ Configures the default Atari output layers. """
    fc0 = nn.Linear(64 * 7 * 7, hidden_size)
    fc0 = nn.utils.spectral_norm(fc0) if spectral_norm else fc0
    fc1 = (
        SharedBiasLinear(hidden_size, out_size)
        if shared_bias
        else nn.Linear(hidden_size, out_size)
    )
    return nn.Sequential(fc0, nn.ReLU(inplace=True), fc1)


def init_weights(module):
    """ Callback for resetting a module's weights to Xavier Uniform and
        biases to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


def no_grad(module):
    """ Callback for turning off the gradient of a module.
    """
    try:
        module.weight.requires_grad = False
    except AttributeError:
        pass


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
        input_ch,
        hist_len,
        action_no,
        hidden_size=256,
        shared_bias=False,
        support=None,
        spectral=None,
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

        # spectral norm configuration
        ft_snorm = spectral in ("conv", "full")
        head_snorm = spectral in ("middle", "full")
        print(
            "Spectral norm:\n\t- convs: {}\n\t- fc0:   {}".format(
                ft_snorm, head_snorm
            )
        )
        # get the feature extractor and fully connected layers
        self.__features = get_feature_extractor(hist_len * input_ch, ft_snorm)
        self.__head = get_head(hidden_size, out_size, shared_bias, head_snorm)

        self.reset_parameters()

    def forward(self, x, probs=False, log_probs=False):
        assert (
            x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor"
        x = x.float().div_(255)
        assert not (
            probs and log_probs
        ), "Can't output both p(s, a) and log(p(s, a))"

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
        return self.__support

    def reset_parameters(self):
        """ Reinitializez parameters to Xavier Uniform for all layers and
            0 bias.
        """
        self.apply(init_weights)

    @property
    def feature_extractor(self):
        return self.__features

    @property
    def head(self):
        return self.__head
