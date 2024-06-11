""" Compares different experience replay sampling methods. """
from functools import partial
from collections import OrderedDict

import torch
from torch.nn import functional as F
import pandas as pd

import gym_fast_envs  # pylint: disable=unused-import
from liftoff.config import read_config, config_to_string

from utils import create_paths, configure_experiment, get_ground_truth


def learn(estimator, transition, loss_fn, optimizer, gamma):
    """ Computes the TD Error and optimises the model.

    Args:
        model (torch.nn.Module): An ensemble model.
        loss_fn (function): The loss functions.
        optimizer (torch.optim.Optim): An optimizer.
        transition (list): A s, a, r_, s_, d_, meta tuple.
        gamma (float): The discount factor.

    Returns:
        torch.tensor: The TD Error.
    """
    state, action, reward, state_, mask, _ = transition

    with torch.no_grad():
        q_targets = estimator(state_)

    qsa = estimator(state).gather(1, action)

    qsa_target = torch.zeros_like(qsa)
    qsa_target[mask] = q_targets.max(1, keepdim=True)[0][mask]

    qsa_target = (qsa_target * gamma) + reward
    loss = loss_fn(qsa, qsa_target)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return loss


def learn_ensemble(ensemble, transition, learner, mem):
    """A wrapper over the `learn` function for optimizing an ensemble of
    estimators (`model`).
    """

    if mem.shuffle:
        transition[5]["boot_mask"] = mask = mem.sample_mask()
    else:
        mask = transition[5]["boot_mask"]

    mids = [mid for mid, m in enumerate(mask) if m != 0]
    for mid in mids:
        # optim.step() is called on all the parameters of the
        # ensemble when we learn a single component of the
        # ensemble. therefore we delete the gradients of all the
        # other estimators to avoid unnecessary operations in
        # optim.step().
        for i, estimator in enumerate(ensemble):
            if i != mid:
                estimator.weight.grad = None

        ensemble_ = partial(ensemble, mid=mid)
        learner(ensemble_, transition)

    return ensemble.var(transition[0], transition[1][0].item())


def train(mem, model, learner, tester, update_cb=None):
    """ The training sequence.
    """
    has_converged = False
    step_cnt = 0

    while not has_converged:
        batch = mem.sample()

        # we are not doing mini-batch learning
        for transition in batch:

            priority = learner(model, transition)

            # update priorities
            if update_cb:
                update_cb(mem, transition, priority)

            step_cnt += 1

            # check for convergence
            has_converged = tester(step_cnt)

            if has_converged:
                break


class Logger:
    def __init__(self, metrics):
        self.__df = pd.DataFrame(columns=metrics)
        self.__idx = 0

    def update(self, loss, step, with_vote=0):
        self.__df.loc[self.__idx] = [step, loss, with_vote]
        self.__idx += 1

    def get_results(self):
        return self.__df


class ConvergenceTester:
    """ Checks for convergence on one or two metrics.
    """

    def __init__(self, model, mdp_size, metrics, log=None, threshold=0.001):
        self.__model = model
        self.__threshold = threshold
        gamma = 1 - 1 / mdp_size
        test_states, optim_q_vals = get_ground_truth(mdp_size, gamma)
        self.__test_states, self.__optim_q_vals = test_states, optim_q_vals
        self.__converged = OrderedDict(
            {metric: {"converged": False, "step_cnt": 0} for metric in metrics}
        )
        if log:
            self.__log = log

    def get_results(self):
        steps = [metric["step_cnt"] for metric in self.__converged.values()]
        return steps, self.__log.get_results()

    def __call__(self, step_cnt):
        return self.__has_converged(step_cnt)

    def __has_converged(self, step_cnt):
        losses = []
        with torch.no_grad():
            y = self.__model(self.__test_states)
            if isinstance(y, tuple):
                for yi in y:
                    losses.append(F.mse_loss(yi, self.__optim_q_vals).item())
            else:
                losses.append(F.mse_loss(y, self.__optim_q_vals).item())

        for i, (k, metric) in enumerate(self.__converged.items()):
            if not metric["converged"]:
                self.__converged[k]["converged"] = losses[i] < self.__threshold
                self.__converged[k]["step_cnt"] = step_cnt

        if self.__log is not None:
            if step_cnt % 100 == 0:
                for i, loss in enumerate(losses):
                    self.__log.update(loss, step_cnt, i)

        return all([m["converged"] for m in self.__converged.values()])


def run(opt):
    """ Experiment trial. """
    lr = 0.25
    learners = [learn, learn_ensemble]

    mem, cb, model, learner, tag = configure_experiment(opt, lr, learners)
    n, mem_size = opt.mdp_size, len(mem)

    # configure tester @ logger
    metrics = ["optim_steps"]
    log_metrics = ["step", "loss", "vote"]
    try:
        if opt.experience_replay.bayesian:
            metrics = ["optim_steps", "vote_optim_steps"]
    except AttributeError:
        pass

    log = Logger(log_metrics)
    tester = ConvergenceTester(model, n, metrics, log=log)

    # run training
    train(mem, model, learner, tester, update_cb=cb)

    # do reporting
    pd.set_option("precision", 4)
    # retrieve logged metrics (optims steps and loss curves)
    optim_steps, results_df = tester.get_results()
    loss = opt.loss if "loss" in opt.__dict__ else "mse"

    # add columns to the loss dataframe curves
    additional_cols = [
        *metrics,
        "N",
        "mem",
        "trial",
        "sampling_type",
        "loss_fn",
    ]
    row_vals = [*optim_steps, n, mem_size, opt.run_id, tag, loss]
    for col, val in zip(additional_cols, row_vals):
        results_df[col] = val

    # save panda
    results_df.to_msgpack(f"./{opt.out_dir}/results.msgpack")

    # log results
    print(config_to_string(opt), flush=True)
    header = f"N={n}, tag={tag}, trial={opt.run_id} results:"
    print("-" * (len(header) + 2))
    print(header, flush=True)
    print("-" * (len(header) + 2))
    print(results_df.head(3), flush=True)


def main():
    """ Entry point. """
    opt = read_config()
    opt = create_paths(opt)
    run(opt)


if __name__ == "__main__":
    main()
