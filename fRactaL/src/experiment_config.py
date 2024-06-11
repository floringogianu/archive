""" Functions that will be common to all.
"""
from functools import reduce

import numpy as np
from torch import optim as O

import src.estimators as E
from src.agents import AGENTS
from src.replay import ExperienceReplay, OfflineExperienceReplay
from src.wrappers import get_env


def get_estimator(opt, env):
    estimator_args = opt.estimator.args
    if opt.estimator.name == "MLP":
        estimator_args["layers"] = [
            reduce(lambda x, y: x * y, env.observation_space.shape),
            *estimator_args["layers"],
        ]
    if opt.estimator.name in ("MinAtarNet", "RandomMinAtarNet"):
        estimator_args["input_ch"] = env.observation_space.shape[-1]
    estimator = getattr(E, opt.estimator.name)(opt.action_cnt, **estimator_args)
    estimator.to(opt.device)

    if (opt.agent.name == "DQN") and ("support" in opt.estimator.args):
        raise ValueError("DQN estimator should not have a support.")
    if (opt.agent.name == "C51") and ("support" not in opt.estimator.args):
        raise ValueError("C51 requires an estimator with support.")

    return estimator


def get_optimizer(opt, estimator):
    for k, v in opt.optim.args.items():
        if isinstance(v, str) and "uniform" in v:
            a, b = [float(x) for x in v.split("(", 1)[1].split(")")[0].split(",")]
            opt.optim.args[k] = np.random.uniform(a, b)

    return getattr(O, opt.optim.name)(estimator.parameters(), **opt.optim.args)


def experiment_config(opt, offline=False):
    """ Configures an environment and an agent """
    env = get_env(opt)
    val_env = get_env(opt, clip_rewards_val=None)
    opt.action_cnt = env.action_space.n

    if offline:
        offline_replays_root = "{}/{}/".format(opt.replay_path, opt.run_id % 5)
        replay = OfflineExperienceReplay(offline_replays_root, False, **opt.replay)
    else:
        # we use the warmup steps in the epsilon-greedy schedule to set the warmup
        # in the replay. If epsilon-greedy is not a schedule the warmup will be
        # given either replay.args.warmup_steps or by batch-size
        if isinstance(opt.agent.args["epsilon"], list):
            opt.replay["warmup_steps"] = opt.agent.args["epsilon"][-1]
        replay = ExperienceReplay(**opt.replay)
        assert (
            replay.warmup_steps == opt.agent.args["epsilon"][-1]
        ), "warmup steps and warmup epsilon should be equal."

    estimator = get_estimator(opt, env)
    optimizer = get_optimizer(opt, estimator)
    policy_evaluation = AGENTS[opt.agent.name]["policy_evaluation"](
        estimator, optimizer, **opt.agent.args
    )
    policy_improvement = AGENTS[opt.agent.name]["policy_improvement"](
        estimator, opt.action_cnt, **opt.agent.args
    )
    return env, val_env, (replay, policy_improvement, policy_evaluation)
