import torch
from itertools import chain, product
import numpy as np


class BernoulliBandit:
    def __init__(self, N=None, p=None, total_trials=100, rng=None) -> None:
        assert N != None or p != None, "Either arms # `N` or probs `p` is required."
        if N != None and p != None:
            assert len(p) == N, "Length of probs `p` should match no. of arms `N`."
        self.N = N
        self.p = torch.rand((N,)) if p is None else p
        self.rng = rng
        self.total_trials = total_trials
        self.q, self.max_a = self.p.max(0)
        self.t = 0
        self._regret = 0
        self.hindsight_regret = 0
        self.correct_action_cnt = 0

    def step(self, act):
        sample = torch.bernoulli(self.p, generator=self.rng)
        r = sample[act]
        # update counter and stats
        self.t += 1
        self._regret += self.q - self.p[act]
        self.hindsight_regret += sample.max() - r
        self.correct_action_cnt += int(act == self.max_a)
        return self.t / self.total_trials, r, self.t == self.total_trials

    def reset(self):
        self.t = 0
        self._regret, self.hindsight_regret, self.correct_action_cnt = 0, 0, 0
        return self.t, False

    @property
    def regret(self):
        return self._regret.item()

    @property
    def hregret(self):
        return self.hindsight_regret.item()

    @property
    def acc(self):
        return self.correct_action_cnt / self.t

    def __repr__(self):
        return "BernoulliBandit(N=%r, trials=%r)" % (self.N, self.total_trials)

    def __str__(self):
        return "BernoulliBandit(N=%r, trials=%r, p=%r)" % (
            self.N,
            self.total_trials,
            self.p,
        )


def get_systematic_tasks(flavor):
    """Pictures :) T=train, V=valid

    Composition         Interpolation       Extrapolation
    T T V V V V V       T T V T T V T       V V V V V V V
    T T V V V V V       V V V V V V V       V V V V V V V
    T T V V V V V       T T V T T V T       T T T T T V V
    T T V V V V V       T T V T T V T       T T T T T V V
    T T V V V V V       V V V V V V V       T T T T T V V
    T T T T T T T       T T V T T V T       T T T T T V V
    T T T T T T T       T T V T T V T       T T T T T V V
    """
    D = 2  # axis of variations (# of arms)
    N = 7  # number of bins on each factor of variation
    p1 = torch.linspace(0, 1, N + 1).unfold(0, 2, 1).tolist()
    p2 = torch.linspace(0, 1, N + 1).unfold(0, 2, 1).tolist()
    tasks = list(product(p1, p2))

    if flavor == "uniform":
        task = [[[0.0, 1.0], [0.0, 1.0]]]
        return task, task
    elif flavor == "random":
        rng = torch.Generator()
        rng.manual_seed(42)
        # take 25 configurations for training, 24 for validation
        trn_idxs = torch.randint(0, N * N, (25,), generator=rng)
        val_idxs = [i for i in range(N * N) if i not in trn_idxs]
    elif flavor == "composition":
        idxs = chain(product([0, 1], range(N)), product(range(N), [0, 1]))
        trn_idxs = np.ravel_multi_index(list(zip(*idxs)), (N, N))
        val_idxs = [i for i in range(N * N) if i not in trn_idxs]
    elif flavor == "interpolation":
        idxs = chain(product([2, 5], range(N)), product(range(N), [2, 5]))
        val_idxs = np.ravel_multi_index(list(zip(*idxs)), (N, N))
        trn_idxs = [i for i in range(N * N) if i not in val_idxs]
    elif flavor == "extrapolation":
        idxs = chain(product([5, 6], range(N)), product(range(N), [5, 6]))
        val_idxs = np.ravel_multi_index(list(zip(*idxs)), (N, N))
        trn_idxs = [i for i in range(N * N) if i not in val_idxs]
    else:
        raise ValueError(f"`{flavor}` is not a flavourous flavor.")

    trn_tasks = [tasks[i] for i in trn_idxs]
    val_tasks = [tasks[i] for i in val_idxs]
    return trn_tasks, val_tasks


class Tasks:
    def __init__(self, split, spec, **task_kws):
        assert split in ("train", "valid", "test"), "Invalid split."
        self.split = split
        self.taskid = -1
        trn, val = get_systematic_tasks(spec)
        self.spec_list = trn if split == "train" else val
        self.task_kws = task_kws

    def get_task(self):
        """Probably this needs to be a class method in the Bandit"""
        # pick a prior for the armed bandit parameters
        spec = self.spec_list[torch.randint(0, len(self.spec_list), (1,))]
        # and use it to sample the parameters of the bandit
        p1 = torch.zeros(1).uniform_(*spec[0])
        p2 = torch.zeros(1).uniform_(*spec[1])
        return BernoulliBandit(p=torch.torch.cat((p1, p2)), **self.task_kws)

    def __iter__(self):
        return self

    def __next__(self):
        self.taskid += 1
        return self.taskid, self.get_task()


def main():
    for tid, bandit in Tasks(10, "test", "composition"):
        print(tid, bandit)


if __name__ == "__main__":
    main()
