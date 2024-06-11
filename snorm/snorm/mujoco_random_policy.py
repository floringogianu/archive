""" Evaluate the random performance of MuJoCo.
"""
import torch
import torch.distributions as D

from muesli import ParallelAverager
from src.common import MUJOCO_ENVS, get_env


def pvalidate(env, policy, valid_ep_cnt=1024):
    """ Validation routine """
    env.eval()
    monitor = ParallelAverager(len(env), "cpu", name="R/ep")

    env.reset()
    while True:
        actions = policy.sample()
        _, reward, done, _ = env.step(actions)
        monitor.put(reward, done)

        if monitor.N >= valid_ep_cnt:
            break
    return monitor.stats()


def main():
    torch.set_printoptions(precision=3)

    B = 32
    raw = []
    for env_name in sorted(MUJOCO_ENVS):
        sim = get_env(env_name, env_no=B, device="cpu")
        N = sim.action_space.shape[-1]
        # policy = D.Uniform(torch.full((B, N), -1.0), torch.full((B, N), 1.0))

        policy = D.TransformedDistribution(
            D.Normal(torch.full((B, N), 0.0), torch.full((B, N), 0.01)),
            [D.transforms.TanhTransform(cache_size=1)],
        )

        val_stats = pvalidate(sim, policy)
        raw.append(val_stats.avgR)

        sim.close()
        del sim

        print("{:>25s}: {:8.2f}".format(env_name, val_stats.avgR))
    print(torch.tensor(raw))


if __name__ == "__main__":
    main()
