""" General utils, including some fancy classes that will be moved
to wintermute if working as intended.
"""
import math
import os
from argparse import Namespace
from collections import defaultdict
from datetime import datetime

import gym
import torch
from termcolor import colored as clr


class StochasticTransitions:
    """ A wrapper over experience replay implementations that adds noise with
    a given precision to the rewards. We use this to simulate a stochastic
    environment within our experimental setup which has a fixed size buffer
    containing the transitions corresponding to an uniform exploration.
    """

    def __init__(self, delegate, noise_variance, device):
        """ StochasticRewards constructor.
        Args:
            delegate (ExperienceReplay): The ER buffer we sample from.
            noise_precision (float): The noise precision.
        """
        self.__delegate = delegate
        self.__noise = torch.distributions.Normal(
            torch.tensor(0.0).to(device),
            torch.tensor(math.sqrt(math.sqrt(noise_variance))).to(device),
        )

    def sample(self):
        """Adds noise to the reward in the sampled transitions.
        Returns:
            list: A batch of transitions.
        """
        batch = self.__delegate.sample()
        if len(batch) == 3:
            # then is a prioritized sampler
            data, idxs, weights = batch
        else:
            data = batch

        # add noise on reward
        data[2] = data[2] + self.__noise.sample(data[2].shape)

        if len(batch) == 3:
            return [data, idxs, weights]
        return data

    def __getattr__(self, name):
        return getattr(self.__delegate, name)

    def __str__(self):
        sigma2 = self.__noise.scale ** 2
        return f"StochasticRewards({str(self.__delegate)}, σ2={sigma2:0.3f})."

    def __len__(self):
        return len(self.__delegate)


class TorchWrapper(gym.ObservationWrapper):
    r""" From numpy to torch. """

    def observation(self, observation):
        r""" Convert from numpy to torch.
        Also change from (h, w, c*hist) to (batch, hist*c, h, w)
        """
        return torch.from_numpy(observation).unsqueeze(0)


class VarianceEstimator:
    r""" Estimate the variance of the target empirically.
    """

    def __init__(self, N, estimator):
        self._estimator = estimator
        self._gamma = 1 - 1 / N
        self._stats = defaultdict(list)

    def __call__(self, batch):
        state, action, reward, state_, mask = batch
        with torch.no_grad():
            mask = mask.squeeze(1)
            qsa_target = torch.zeros(state.shape[0], 1, device=state.device)
            if state_.nelement():
                qvals_ = self._estimator(state_)
                qsa_target[mask] = (
                    qvals_.max(1, keepdim=True)[0] * self._gamma
                ) + reward
            else:
                qsa_target += reward

            key = self._get_key(state, action)
            self._stats[key].append(qsa_target.cpu().squeeze().item())

    def _get_key(self, state, action):
        state_idx = state.squeeze().nonzero().squeeze().item()
        return f"s{state_idx}a{action.squeeze().item()}"

    def __str__(self):
        rep = "\nTarget Variance estimation:\n"
        for i, (k, v) in enumerate(sorted(self._stats.items())):
            v = torch.tensor(v)
            if (i + 1) % 2 == 0:
                rep += (
                    f"{v.var():2.4f}   |  {len(_v):3d}/{len(v):3d} samples.\n"
                )
            else:
                _v = v
                rep += f"{k[:2]}:  {v.var():2.4f}    "

        rep += "\nTarget Mean estimation:\n"
        for i, (k, v) in enumerate(sorted(self._stats.items())):
            v = torch.tensor(v)
            if (i + 1) % 2 == 0:
                rep += f"{v.mean():2.4f}\n"
            else:
                rep += f"{k[:2]}:  {v.mean():2.4f}    "

        return rep


class ConvergenceTester:
    r""" Decides if the algorithm reached the ground-truth Q-values.

    Args:
        opt (Namespace): Experiment config.
        estimator (nn.Module): The value function estimator.
    """

    def __init__(self, N, threshold, estimator, device):
        self._estimator = estimator
        self._states = torch.eye(N, dtype=torch.uint8).to(device)
        self._ground_truth = torch.zeros(N, 2).to(device)
        self._threshold = threshold

        gamma = 1 - 1 / N
        qsa_star = [gamma ** i for i in range(N - 1, -1, -1)]
        for i, qsa in enumerate(qsa_star):
            act_idx = 0 if i % 2 == 0 else 1
            self._ground_truth[i][act_idx] = qsa

    def test(self):
        r""" Tests for convergence. Return decison and error. """
        with torch.no_grad():
            qvals = self._estimator(self._states)
        err = torch.nn.functional.mse_loss(qvals, self._ground_truth).item()

        return err < self._threshold, err

    def get_qvals(self):
        r""" Return a table with the Q-values for all the states. """
        with torch.no_grad():
            qvals = self._estimator(self._states).cpu().numpy()
        qstar = self._ground_truth.cpu().numpy()
        header = (
            "\n    -----------------------------------------\n"
            + "           Q(s,a)        |       Q*(s,a)     \n"
            + "    -----------------------------------------\n"
        )
        rows = []
        for qsa, qsa_star in zip(qvals, qstar):
            qsa = [f"{q:>6.3f}" for q in qsa]
            qsa_star = [f"{q:>6.3f}" for q in qsa_star]
            rows.append(
                "    " + "    ".join(qsa) + "    |   " + "   ".join(qsa_star)
            )
        return header + "\n".join(rows)


def create_paths(args: Namespace) -> Namespace:
    """ Creates directories containing the results of the experiments when you
        use `python main.py` instead of `liftoff main.py`. It uses liftoff
        convention for the folder names: `timestamp_experimentname`.
    """
    time_stamp = "{:%Y%b%d-%H%M%S}".format(datetime.now())
    print(time_stamp)
    if not hasattr(args, "out_dir") or args.out_dir is None:
        if not os.path.isdir("./results"):
            os.mkdir("./results")
        out_dir = f"./results/{time_stamp}_{args.experiment:s}"
        os.mkdir(out_dir)
        args.out_dir = out_dir
    elif not os.path.isdir(args.out_dir):
        raise Exception(f"Directory {args.out_dir} does not exist.")

    if not hasattr(args, "run_id"):
        args.run_id = 0

    return args


def config_to_string(
    cfg: Namespace, indent: int = 0, color: bool = True
) -> str:
    """Creates a multi-line string with the contents of @cfg."""

    text = ""
    for key, value in cfg.__dict__.items():
        ckey = clr(key, "yellow", attrs=["bold"]) if color else key
        text += " " * indent + ckey + ": "
        if isinstance(value, Namespace):
            text += "\n" + config_to_string(value, indent + 2, color=color)
        else:
            cvalue = clr(str(value), "white") if color else str(value)
            text += cvalue + "\n"
    return text
