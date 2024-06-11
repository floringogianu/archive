""" Offline script for various stuff such as empirically checking the
    Lipschitz constant or re-evalutating the models.
"""
from copy import deepcopy
from pathlib import Path

import rlog
import torch
from liftoff import parse_opts

import src.io_utils as ioutil
from src.training import get_estimator
from src.wrappers import get_env
from src.agents import AGENTS


def compute_action_gap(pi):
    vals, _ = pi.full.flatten().topk(k=2)
    return vals[0] - vals[1]


def sample_env(policy, env, episodes):
    """ Validation routine """
    policy.estimator.eval()

    step = 0
    for ep_cnt in range(episodes):
        obs, done = env.reset(), False
        while not done:
            with torch.no_grad():
                obs = obs.float()
                pi = policy.act(obs)

            obs, reward, done, _ = env.step(pi.action)
            rlog.put(
                action_gap=compute_action_gap(pi),
                qvals=pi.full,
                reward=reward,
                done=done,
                ep_cnt=ep_cnt,
            )
            step += 1
    rlog.traceAndLog(step=step)


def load_policy(env, ckpt_path, opt):
    opt.action_cnt = env.action_space.n
    estimator = get_estimator(opt, env)
    agent_args = opt.agent.args
    agent_args["epsilon"] = 0.0  # purely max
    policy = AGENTS[opt.agent.name]["policy_improvement"](
        estimator, opt.action_cnt, **agent_args
    )
    idx = int(ckpt_path.stem.split("_")[1])
    rlog.info(f"Loading {ckpt_path.stem}")
    ckpt = ioutil.load_checkpoint(
        ckpt_path.parent, idx=idx, verbose=False, device=torch.device(opt.device)
    )

    if opt.estimator.args["spectral"] is not None:
        ioutil.special_conv_uv_buffer_fix(policy.estimator, ckpt["estimator_state"])
    policy.estimator.load_state_dict(ckpt["estimator_state"])
    return policy, idx


# results/experiment/variation/0
def run(opt):
    """ Entry point of the experiment """

    # we only do this for the noiseless experiments
    if opt.demon.name is not None:
        return

    # this is a bit of a hack, it would be nice to change it
    # when launching the experiment. It generally only affects the logger.
    if "AG" not in opt.experiment:
        opt.experiment += "--AG"

    rlog.init(opt.experiment, path=opt.out_dir, relative_time=True)
    rlog.addMetrics(
        rlog.ValueMetric("action_gap", metargs=["action_gap"]),
        rlog.ValueMetric("qvals", metargs=["qvals"]),
        rlog.ValueMetric("ep_cnt", metargs=["ep_cnt"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
    )

    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(opt.out_dir)
    ckpt_paths = sorted(root.glob("**/checkpoint*"))

    rlog.info("Begin measuring the action gap.")
    rlog.info("Runing experiment on {}.".format(opt.device))
    rlog.info("Found {:3d} checkpoints.".format(len(ckpt_paths)))

    if (Path(opt.out_dir) / "max_ckpt").exists():
        ckpt_paths = [
            p
            for p in ckpt_paths
            if int(p.stem.split("_")[1]) == int((root / "max_ckpt").read_text())
        ]
        rlog.info("IMPORTANT! Found max_ckpt @{}.".format(ckpt_paths[0]))
    else:
        raise ValueError(f"No max_ckpt found in {opt.out_dir}.")

    for ckpt_path in ckpt_paths:
        env = get_env(opt, mode="testing")
        policy, step = load_policy(env, ckpt_path, deepcopy(opt))

        sample_env(policy, env, 100)


def main():
    """ Liftoff
    """
    run(parse_opts())


if __name__ == "__main__":
    main()
