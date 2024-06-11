"""
This file implements everything needed for SVI and sampling from the posterior
distributions. The boilerplate code is taken from 
https://github.com/floringogianu/schrodinger and adapted such that it 
implements the networks from "Good Initializations of Variational Bayes for 
Deep Models": https://arxiv.org/pdf/1810.08083.pdf
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import rlog


def kl_div(μ, σ):
    """ Computes the KL divergence between a normal multivariate and a
        diagonal normal distribution.
    Args:
        post (torch.distribution.Normal): A normal distribution.
    Returns:
        torch.tensor: The KL Divergence
    """
    return (μ.pow(2) + σ.exp() - 1 - σ).sum() * 0.5


class SVIModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self._mle_model = model
        self._posterior = OrderedDict()
        self._prior = OrderedDict()

        self.softplus = nn.Softplus()

        self.predictions = None
        self._normals = None

        for weight_name, weight in model.named_parameters():
            # register the distribution and its parameters
            self._posterior[weight_name] = {
                "loc": nn.Parameter(torch.randn_like(weight) * 0.1),
                "logvar": nn.Parameter(torch.ones_like(weight).fill_(-5)),
            }
            # register the priors. Actually not needed because we have a
            # special case for computing the KL.
            self._prior[weight_name] = Normal(
                loc=torch.zeros_like(weight), scale=torch.ones_like(weight)
            )

    def set_freq_prior(self, freq_model):
        for weight_name, weight in freq_model.named_parameters():
            self._prior[weight_name] = Normal(
                loc=weight, scale=torch.ones_like(weight)
            )

    def set_posterior(self):
        self._normals = {
            weight_name: Normal(
                loc=posterior["loc"], scale=torch.exp(0.5 * posterior["logvar"])
            )
            for weight_name, posterior in self._posterior.items()
        }

    def forward(self, x, M=1, full=False):
        out = torch.stack([self._forward(x) for _ in range(M)])
        if full:
            return out
        return out.mean(0)

    def _forward(self, x):
        """ This is a hackish forward in which we are reconstructing the
        forward operations in `self._mle_model`. Ideally this function would
        only:
            1. draw samples from the posterior distribution using the
            reparametrization trick.
            2. perform inference with the **original** model and the sampled
            weights.
        """

        #         for _, posterior in self._posterior.items():
        #             posterior.scale = self.softplus(posterior.scale)

        # draw samples from the posterior distribution

        if self._normals is None:
            self.set_posterior()

        posterior_sample = {
            weight_name: self._normals[weight_name].rsample()
            for weight_name, posterior in self._posterior.items()
        }

        # do a hackish forward through the network
        for layer_name, layer in self._mle_model.named_children():
            if isinstance(layer, nn.Linear):
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)

                x = F.linear(
                    x,
                    posterior_sample[f"{layer_name}.weight"],
                    posterior_sample[f"{layer_name}.bias"],
                )
            elif isinstance(layer, nn.Conv2d):
                x = F.conv2d(
                    x,
                    posterior_sample[f"{layer_name}.weight"],
                    posterior_sample[f"{layer_name}.bias"],
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    layer.groups,
                )
            elif isinstance(layer, nn.MaxPool2d):
                x = F.max_pool2d(
                    x,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    layer.ceil_mode,
                    layer.return_indices,
                )
            elif isinstance(layer, nn.LogSoftmax):
                x = F.log_softmax(x, layer.dim)
            else:
                print(layer)
                raise TypeError("Don't know what type of layer this is.")

            if layer_name not in ["out", "out_activ"]:
                x = torch.relu(x)

        return x

    def get_kl_div(self):
        """ Return the KL divergence for the complete posterior distribution.
        """
        kls = [
            kl_div(posterior["loc"], torch.exp(0.5 * posterior["logvar"]))
            for posterior in self._posterior.values()
        ]
        return torch.stack(kls).sum()

    def std(self):
        """ Return the standard deviations of the posterior distribution.
        """
        return [
            torch.exp(0.5 * p["logvar"].data).cpu().numpy().ravel()
            for p in self._posterior.values()
        ]

    def var(self):
        """ Return the variances of the posterior distribution.
        """
        return [
            torch.exp(p["logvar"].data).cpu().numpy().ravel()
            for p in self._posterior.values()
        ]

    def mu(self):
        """ Return the mean for the complete posterior distribution.
        """
        return [
            p["loc"].data.cpu().numpy().ravel()
            for p in self._posterior.values()
        ]

    def parameters(self):
        """ Returns the variational parameters of the posterior distribution.
        """
        params = []
        for dist in self._posterior.values():
            params.append(dist["loc"])
            params.append(dist["logvar"])
        return params

    def get_predictive_variance(self, regression=False):
        """ Returns the predictive variance for the result of the last call to
        eval_forward
        """
        probabilities = self.predictions.exp()

        sigma2 = probabilities.var(0)

        if not regression:
            y_hat = probabilities.mean(0).max(1)[1]
            sigma2 = sigma2.gather(1, y_hat.unsqueeze(1))

        return sigma2

    def sync_mle_model(self):
        """ Uses the mean of the posterior to set the equivalent MLE model.
        """
        state_dict = {k: p["loc"] for k, p in self._posterior.items()}
        self._mle_model.load_state_dict(state_dict)
