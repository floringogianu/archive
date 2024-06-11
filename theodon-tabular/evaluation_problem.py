r""" Script for solving the values estimation problem on a fixed experience
replay buffer.
"""
import itertools
import random
from functools import partial
import torch
from torch import optim
import gym
import gym_classic  # pylint: disable=unused-import
import rlog
from liftoff import parse_opts
from wintermute.replay import ExperienceReplay
from wintermute.policy_improvement import DQNPolicyImprovement, get_dqn_loss
from src.models import get_estimator
from src.utils import (
    ConvergenceTester,
    TorchWrapper,
    StochasticTransitions,
    config_to_string,
)


def priority_update(mem, idxs, weights, dqn_loss, variances=None):
    """ Callback for updating priorities in the proportional-based experience
    replay and for computing the importance sampling corrected loss.
    """
    losses = dqn_loss.loss
    with torch.no_grad():
        td_errors = (dqn_loss.qsa_targets - dqn_loss.qsa).detach().abs()

    if variances:
        mem.update(idxs, [var.item() for var in variances.detach()])
        return (losses * weights.to(losses.device).view_as(losses)).mean()
    mem.update(idxs, [td.item() for td in td_errors])
    return (losses * weights.to(losses.device).view_as(losses)).mean()


def learn_ensemble(opt, experience_replay, ensemble, optimizer, loss_fn):
    """ Learning routine for an ensemble.
    """
    step_cnt, has_converged = 0, False
    boot_prob, boot_no = opt.mask_prob, opt.estimator.ensemble

    train_log = rlog.getLogger(opt.experiment + ".train")
    convergence_tester = ConvergenceTester(
        opt.N, opt.convergence_threshold, ensemble, opt.device
    )
    bootmask_distribution = torch.distributions.Bernoulli(boot_prob)

    # learning loop
    while not (has_converged or step_cnt == float(opt.max_steps)):

        batch = experience_replay.sample()
        if len(batch) == 3:
            # then is a prioritized sampler
            batch, idxs, weights = batch

        # mask the given transition and train on a subset of the ensemble
        mids = bootmask_distribution.sample((boot_no,)).nonzero().numpy()
        for mid in mids.flatten():
            dqn_loss = get_dqn_loss(
                batch,
                partial(ensemble, mid=mid),
                gamma=opt.gamma,
                loss_fn=loss_fn,
            )

            dqn_loss.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # compute priority
        if len(mids):  # pylint: disable=len-as-condition
            if opt.priority == "td":
                with torch.no_grad():
                    dqn_loss = get_dqn_loss(
                        batch, ensemble, gamma=opt.gamma, loss_fn=loss_fn
                    )
                    priority_update(experience_replay, idxs, weights, dqn_loss)
            elif opt.priority == "variance":
                with torch.no_grad():
                    priority_update(
                        experience_replay,
                        idxs,
                        weights,
                        dqn_loss,
                        variances=ensemble.var(batch[0], action=batch[1]),
                    )
            else:
                pass

        # logging
        has_converged, err = convergence_tester.test()
        train_log.put(err=err)
        step_cnt += 1
        if step_cnt % 1000 == 0:
            summary = train_log.summarize()
            for_info = {k: v for k, v in summary.items() if k != "mse_err"}
            train_log.info(
                "@{0:5d}, avg_err={avg_err:2.4f}".format(step_cnt, **for_info)
            )
            train_log.trace(step=step_cnt, **summary)
            train_log.reset()

        if has_converged:
            train_log.trace(step=step_cnt, **train_log.summarize())

    status = "Converged" if has_converged else "Converged NOT"
    train_log.info("%s @ %s, err=%2.4f.", status, step_cnt, err)
    train_log.trace(step=step_cnt, converged=int(has_converged))
    train_log.info(convergence_tester.get_qvals())


def learn(opt, experience_replay, policy_improvement):
    """ Sample from experience replay and learn. Also tracks and logs values.
    """
    train_log = rlog.getLogger(opt.experiment + ".train")
    convergence_tester = ConvergenceTester(
        opt.N,
        opt.convergence_threshold,
        policy_improvement.estimator,
        opt.device,
    )

    step_cnt, has_converged = 0, False
    while not (has_converged or step_cnt == float(opt.max_steps)):

        # sample data
        batch = experience_replay.sample()
        if len(batch) == 3:
            # this is a prioritized sampler
            batch, idxs, weights = batch
            clbk = partial(priority_update, experience_replay, idxs, weights)
        else:
            clbk = None

        # learn
        policy_improvement(batch, cb=clbk)

        # logging
        has_converged, err = convergence_tester.test()
        train_log.put(err=err)
        step_cnt += 1
        if step_cnt % 1000 == 0:
            summary = train_log.summarize()
            for_info = {k: v for k, v in summary.items() if k != "mse_err"}
            train_log.info(
                "@{0:5d}, avg_err={avg_err:2.4f}".format(step_cnt, **for_info)
            )
            train_log.trace(step=step_cnt, **summary)
            train_log.reset()

        if has_converged:
            train_log.trace(step=step_cnt, **train_log.summarize())

    status = "Converged" if has_converged else "Converged NOT"
    train_log.info("%s @ %s, err=%2.4f.", status, step_cnt, err)
    train_log.trace(step=step_cnt, converged=int(has_converged))
    train_log.info(convergence_tester.get_qvals())


def fill_experience_replay_(env, experience_replay, device):
    """ Sample all the transitions equivalent to a random policy sampling all
        the possible sequence of actions leading to termination of the MDP push
        them in the experience replay.
        There are:
            - 2^N seqences of possible actions
            - 2^(N+1)-2 transitions for all the sequences
    """
    actions = [0, 1]
    N = env.observation_space.shape[1]

    # Generate all possible sequence of actions and randomize them
    action_sequences = list(itertools.product(*[actions] * N))
    random.shuffle(action_sequences)

    for act_seq in action_sequences:
        state_, done = env.reset(), False

        for action in act_seq:
            state = state_.clone().to(device)
            state_, reward, done, _ = env.step(action)
            experience_replay.push((state, action, reward, state_, done))
            if done:
                break


def augment_options_(opt):
    """ Some options need to be set at runtime.
    """
    opt.gamma = 1 - 1 / opt.N
    opt.er.capacity = 2 ** (opt.N + 1) - 2
    opt.device = torch.device("cuda") if opt.cuda else torch.device("cpu")


def run(opt):
    r""" Function executed by `liftoff`.
    """
    augment_options_(opt)

    # configure loggers
    rlog.init(opt.experiment, path=opt.out_dir)
    rlog.info(
        "Starting experiment using the following settings:\n%s",
        config_to_string(opt),
    )
    train_log = rlog.getLogger(opt.experiment + ".train")
    train_log.addMetrics(
        [
            rlog.ValueMetric("mse_err", metargs=["err"]),
            rlog.AvgMetric("avg_err", metargs=["err", 1]),
        ]
    )

    # configure experience replay and fill it
    env = TorchWrapper(gym.make(f"BlindCliffWalk-N{opt.N}-v0"))
    experience_replay = ExperienceReplay(**opt.er.__dict__)()
    fill_experience_replay_(env, experience_replay, opt.device)

    if opt.noise_variance:
        experience_replay = StochasticTransitions(
            experience_replay, opt.noise_variance, opt.device
        )

    # configure estimator and optimizer
    estimator = get_estimator(opt)

    rlog.info("\n   er:%s\n   ent:%s\n", experience_replay, estimator)

    # do some learning
    if opt.estimator.ensemble:
        optimizer = optim.SGD(estimator.parameters(), lr=opt.lr)
        loss_fn = getattr(torch.nn, opt.loss_fn)(reduction="none")
        learn_ensemble(opt, experience_replay, estimator, optimizer, loss_fn)
    else:
        optimizer = optim.SGD(estimator.parameters(), lr=opt.lr)
        learn(
            opt,
            experience_replay,
            DQNPolicyImprovement(
                estimator,
                optimizer,
                opt.gamma,
                target_estimator=False,
                loss_fn=opt.loss_fn,
            ),
        )


def main():
    r""" Entry point of the program when executing directly with python.
    """

    # read config files using liftoff
    opt = parse_opts()
    # run your experiment
    run(opt)


if __name__ == "__main__":
    main()
