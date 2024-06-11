""" Entry point. """
import torch.optim as optim
import numpy as np

import gym
import gym_classic  # pylint: disable=W0611

from wintermute.replay import ExperienceReplay
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_improvement import DQNPolicyImprovement

import rlog
from liftoff import parse_opts
from src.utils import TorchWrapper, ConvergenceTester, config_to_string
from src.rl_routines import Episode, DQNPolicy
from src.models import Estimator


def policy_iteration(opt, env, policy):
    r""" Policy improvement routine.

    Args:
        opt (Namespace): Configuration values.
        env (gym.env): The game we are training on.
        policy (DQNPolicy): A DQN agent.
        log (Logger): A simple logger.
    """
    train_log = rlog.getLogger(opt.experiment + ".train")
    convergence_tester = ConvergenceTester(
        opt.N, opt.convergence_threshold, policy.policy_improvement.estimator
    )
    has_converged = False

    while not (has_converged or policy.steps == opt.max_steps):

        # play one episode
        for _state, _action, reward, state, done in Episode(env, policy):

            # push to memory
            policy.push((_state, _action, reward, state, done))

            # learn
            if policy.steps >= opt.er.capacity:
                policy.learn()

                if opt.update_target_freq:
                    if policy.steps % opt.update_target_freq == 0:
                        policy.policy_improvement.update_target_estimator()

            # test convergence
            has_converged, err = convergence_tester.test()

            # logging
            train_log.put(reward=reward, done=done, err=err)

            if policy.steps % 1000 == 0:
                summary = train_log.summarize()
                train_log.info(
                    {k: v for k, v in summary.items() if k != "mse_err"}
                )
                train_log.trace(step=policy.steps, **summary)
                train_log.reset()

            if has_converged:
                train_log.trace(step=policy.steps, **train_log.summarize())
                break

    if has_converged:
        train_log.info("Converged @ %s, err=%2.3f.", policy.steps, err)
        train_log.info(convergence_tester.get_qvals())
    else:
        train_log.info("Converged NOT after %s, err=%2.3f.", policy.steps, err)
        train_log.info(convergence_tester.get_qvals())


def run(opt):
    """ Where the action really happens. """
    np.set_printoptions(suppress=True)

    # aditional options
    states_no = 2 ** (opt.N + 1) - 2
    opt.er.capacity = states_no
    opt.exploration.steps = states_no
    opt.exploration.warmup_steps = states_no
    if not hasattr(opt, "experiment"):
        opt.experiment = f"dqn_N{opt.N}"

    # configure loggers
    rlog.init(opt.experiment, path=opt.out_dir)
    rlog.info("Starting experiment using the following settings:")
    rlog.info(config_to_string(opt))

    train_log = rlog.getLogger(opt.experiment + ".train")
    train_log.addMetrics(
        [
            rlog.AvgMetric("rw_per_ep", metargs=["reward", "done"]),
            rlog.ValueMetric("mse_err", metargs=["err"]),
            rlog.FPSMetric("train_fps", metargs=[1]),
        ]
    )

    # configure algorithm objects
    env = TorchWrapper(gym.make(f"BlindCliffWalk-N{opt.N}-v0"))
    gamma = 1 - 1 / opt.N
    estimator = Estimator(opt.N * opt.er.hist_len, 2, bias=opt.bias)
    estimator.reset_parameters()

    policy = DQNPolicy(
        EpsilonGreedyPolicy(
            estimator, env.action_space.n, opt.exploration.__dict__
        ),
        DQNPolicyImprovement(
            estimator,
            optim.SGD(estimator.parameters(), lr=opt.lr),
            gamma,
            target_estimator=False if opt.update_target_freq == 0 else None,
            loss_fn=opt.loss_fn,
        ),
        ExperienceReplay(**opt.er.__dict__)(),
    )

    # do the training
    policy_iteration(opt, env, policy)


def main():
    r""" Entry point of the program."""

    # read config files using liftoff
    opt = parse_opts()
    # run your experiment
    run(opt)


if __name__ == "__main__":
    main()
