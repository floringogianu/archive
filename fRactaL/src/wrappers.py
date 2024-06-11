""" A torch wrapper and an utility function. 
"""
import gym
import numpy as np
import torch

from src.envs import ALE, MinAtar

__all__ = [
    "get_env",
    "TorchWrapper",
]


class TorchWrapper(gym.ObservationWrapper):
    """ Receives numpy arrays and returns torch tensors.
        Used with robotics envs.
    """

    def __init__(self, env, device):
        super().__init__(env)
        self._device = device

    def observation(self, observation):
        obs = torch.from_numpy(observation)
        if obs.ndim == 3:
            obs = obs.permute(2, 0, 1).byte().to(self._device)
            return obs.view(1, 1, *obs.shape)
        return obs.view(1, 1, -1).float().to(self._device)


def get_env(opt, **kwargs):
    """ Configures an environment based on the name and options. """
    if opt.game.split("-")[0] in ("LunarLander", "CartPole"):
        return TorchWrapper(gym.make(opt.game), opt.device)
    if opt.game.split("-")[0] == "MinAtar":
        return TorchWrapper(MinAtar(opt.game.split("-")[-1], **kwargs), opt.device)
    # probably an ALE game
    return ALE(
        opt.game,
        np.random.randint(100_000),
        opt.device,
        clip_rewards_val=kwargs.get("clip_rewards_val", 1),
    )


if __name__ == "__main__":
    pass
