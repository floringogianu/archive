import torch
import torch.distributions as D
import torch.nn.functional as F


class TanhNormal:
    """A Gaussian policy transformed by a Tanh function."""

    def __init__(self, action_num, min_std=0.0005, init_std=0) -> None:
        self.action_num = action_num
        self.min_std = min_std
        # find a constant such that the initial scale to be close to 0.5
        # c = softplus^-1(wanted_std - min_std)
        self.c_rho = torch.tensor(0.5 - min_std).expm1().log() if init_std else 0.0

    def __call__(self, x):
        loc, scale = torch.split(x, self.action_num, -1)

        # print(f'mean_loc=l{loc.mean()}, mean_scale={scale.mean()}')
        if self.c_rho:
            scale = F.softplus(scale - self.c_rho) + self.min_std
        else:
            scale = F.softplus(scale) + self.min_std

        normal = D.Normal(loc, scale)
        pi = D.TransformedDistribution(
            normal, [D.transforms.TanhTransform(cache_size=1)]
        )
        return pi, normal.entropy()

    def __repr__(self) -> str:
        return "tanh(Normal(loc: {a}, scale: {a}))".format(a=("B", self.action_num))
