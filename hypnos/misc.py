""" Where the AGI happens.
"""
from argparse import Namespace

import psutil
from rich import print

from src.envs import get_env
from src.replay import OfflineExperienceReplay


def main():
    for game in ["asterix", "breakout", "pong", "riverraid", "seaquest", "space_invaders"]:
        opt = Namespace(game=game, stochasticity="random_starts", device="cpu")
        print(f"{game:12s}", get_env(opt).action_space)


def replay_check():
    replay = OfflineExperienceReplay(
        "/atari/Asterix/1/replay_logs", True, capacity=1000000, device="cpu"
    )
    for _ in range(10):
        replay.reload_N(N=5, workers=5)
        mem = psutil.virtual_memory()
        print(
            "{:10s}: {:6.1f}GB\n{:10s}: {:6.1f}".format(
                "Used", mem.used / (1024 ** 2), "Active", mem.active / (1024 ** 2)
            )
        )


if __name__ == "__main__":
    main()
