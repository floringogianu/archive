""" Experience Replay implementations.

    WARNING: these implementations have not yet been tested as RingBuffers as
    they are usually used in RL algorithms.

    1. GreedyHeapqSampler implements greedy sampling from the buffer using
    python's standard library `heapq`.

    2. GreedyPQSampler also implements greedy sampling but using the
    PriorityQueue in `data_structures.py`. The two implementations perform the
    same.

    3. RankSampler implements the rank-based prioritization using the
    PriorityQueue in `data_structures.py`.

    4. ProportionalSampler implements the proportional based prioritization
    using the SumTree in `data_structures.py`.

    5. BootstrappedER is a wrapper over any ExperienceReplay implementation.
    Its purpose is to additionally implement data bootstrapping.
"""
import heapq

import torch
import numpy as np

from wintermute.data_structures import NaiveExperienceReplay
from data_structures import PriorityQueue
from data_structures import SumTree


def get_experience_replay(capacity, sampling="uniform", batch_size=1, **kwargs):
    """ Factory for various Experience Replay implementations. """

    # common Experience Replay args
    batch_size = capacity if batch_size > capacity else batch_size
    er_args = {"capacity": capacity, "batch_size": batch_size}

    # additional args depending on implementation
    if sampling == "uniform":
        er_args["collate"] = _collate
        er_args["full_transition"] = True
    elif sampling in ("rank", "proportional"):
        if "alpha" in kwargs:
            er_args["alpha"] = kwargs["alpha"]

    # pick callback used for updating priorities
    cb = greedy_update if sampling in ("greedy-pq", "greedy-hpq") else None
    cb = stochastic_update if sampling in ("rank", "proportional") else cb

    # bootstrapping
    if "bayesian" in kwargs:
        if kwargs["bayesian"]:
            mem = BootstrappedER(
                sampling,
                er_args,
                kwargs["boot_p"],
                kwargs["boot_no"],
                kwargs["boot_shuffle"],
            )
            return mem, cb

    return BUFFERS[sampling](**er_args), cb


def greedy_update(mem, transition, loss):
    """ Callback for reinserting transitions in the experience replay with the
    new priorities.
    """
    td_error = loss.detach().abs().item()
    mem.push_updated(td_error, torch2numpy(transition))


def stochastic_update(mem, transition, loss):
    """ Callback for updating priorities in the rank-based experience replay.
    """
    td_error = loss.detach().abs().item()
    idx = transition[5]["idx"]
    mem.update(idx, td_error)


def torch2numpy(transition):
    """ Convert a torch transition to a numpy one.
    """
    s, a, r, s_, mask, meta = transition
    return (s.numpy(), a.item(), r.item(), s_.numpy(), 1 - mask.item(), meta)


def _collate(data):
    batch = [
        (
            torch.from_numpy(s).float(),
            torch.LongTensor([a]).unsqueeze(1),
            torch.FloatTensor([r]).unsqueeze(1),
            torch.from_numpy(s_).float(),
            1 - torch.ByteTensor([d]).unsqueeze(1),
            meta,
        )
        for s, a, r, s_, d, meta in data
    ]
    return batch


def _collate_with_index(data):
    # TODO: FIX THIS!!!
    batch = [
        (
            torch.from_numpy(s).float(),
            torch.LongTensor([a]).unsqueeze(1),
            torch.FloatTensor([r]).unsqueeze(1),
            torch.from_numpy(s_).float(),
            1 - torch.ByteTensor([d]).unsqueeze(1),
            {**meta, "idx": i},
        )
        for i, (s, a, r, s_, d, meta) in data
    ]
    return batch


class GreedyHeapqSampler:
    """ Implements the greedy TD-error sampling technique in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) used in the
        BlindCliffWalk experiments.
    """

    def __init__(self, capacity, batch_size=1, collate=None):
        self.__memory = []
        self.__batch_size = batch_size
        self.__capacity = capacity

        self.__collate = collate or _collate
        self.__position = 0

    def push(self, transition, priority=None):
        """ Adds a transition in the priority queue.

            The first element in the item is used in the comparison operations
            in the heapq. Having no initial td errors we populate them with a
            very small value so that we make sure they get sampled. We will
            gradually update the priorities.

            Second element is used by `heapq` to break eventual ties in
            priority and for the first optimization sweep will act as the
            actual prioriy.
        """
        priority = (self.__position + 1000) or priority
        heapq.heappush(self.__memory, (-priority, transition))
        self.__position = (self.__position + 1) % self.__capacity

    def sample(self):
        """ Returns the transitions with the highest priority. """
        samples = [
            heapq.heappop(self.__memory)[1] for _ in range(self.__batch_size)
        ]
        return self.__collate(samples)

    def push_updated(self, td_err, transition):
        """ Adds a transition prioritized by the TD-Error. """
        td_err += np.random.rand() * 1e-7
        heapq.heappush(self.__memory, (-td_err, transition))

    def __len__(self):
        return len(self.__memory)

    def __repr__(self):
        return f"GreedyHeapqSampler(size={len(self)})"


class GreedyPQSampler:
    """ Implements the greedy TD-error sampling technique in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) used in the
        BlindCliffWalk experiments.
    """

    def __init__(self, capacity, batch_size=1, collate=None):
        self.__pq = PriorityQueue()
        self.__batch_size = batch_size
        self.__capacity = capacity

        self.__collate = collate or _collate
        self.__position = 0

    def push(self, transition, priority=None):
        priority = (self.__position + 1000) or priority
        self.__pq.push((-priority, transition))
        self.__position = (self.__position + 1) % self.__capacity

    def sample(self):
        samples = [self.__pq.pop()[1] for _ in range(self.__batch_size)]
        return self.__collate(samples)

    def push_updated(self, td_err, transition):
        td_err += np.random.rand() * 1e-7
        self.__pq.push((-td_err, transition))

    def __len__(self):
        return len(self.__pq)

    def __repr__(self):
        return f"GreedyPQSampler(size={len(self)})"


class RankSampler:
    """ Implements the rank-based sampling technique in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) used in the
        BlindCliffWalk experiments.
    """

    def __init__(self, capacity, batch_size=1, collate=None, alpha=0.9):
        self.__pq = PriorityQueue()
        self.__capacity = capacity
        self.__batch_size = batch_size

        self.__collate = collate or _collate_with_index
        self.__position = 0

        self.__alpha = alpha
        self.__partitions = []
        self.__segments = []
        self.__segment_probs = []

    def push(self, transition, priority=None):
        """ Commit new transition to the PQ. If priority is not available then
        initialize with a large value making sure every new transition is being
        sampled and updated. Since our PQ is a Min-PQ, we use the negative of
        the priority.
        """
        priority = (self.__position + 1000) or priority
        self.__pq.push((-priority, transition))
        self.__position = (self.__position + 1) % self.__capacity

        if self.__capacity == len(self):
            self.__compute_segments()

    def sample(self):
        # TODO: docstring
        segment_idxs = np.random.choice(
            len(self.__segments), size=self.__batch_size, p=self.__segment_probs
        )
        segments = [self.__segments[sid] for sid in segment_idxs]
        idxs = [np.random.randint(*segment) for segment in segments]

        # warning, atypical use of a priority queue
        # pylint: disable=protected-access
        samples = [(i, self.__pq._PriorityQueue__heap[i][1]) for i in idxs]
        # pylint: enable=protected-access

        return self.__collate(samples)

    def update(self, idx, priority):
        self.__pq.update(idx, -priority)

    def sort(self):
        for _ in range(len(self)):
            self.__pq.push(self.__pq.pop())

    def __compute_segments(self):
        N = len(self)
        self.__partitions = []
        self.__segments = []

        segment_sz = int(np.floor(N / self.__batch_size))
        for i in range(self.__batch_size):
            a = i * segment_sz
            b = (i + 1) * segment_sz if i != (self.__batch_size - 1) else N

            partition = [(1 / (idx + 1)) ** self.__alpha for idx in range(a, b)]

            self.__partitions.append(np.sum(partition))
            self.__segments.append((a, b))

        self.__segment_probs = [
            p / sum(self.__partitions) for p in self.__partitions
        ]

    def __len__(self):
        return len(self.__pq)

    def __repr__(self):
        props = f"size={len(self)}, α={self.__alpha}, batch={self.__batch_size}"
        return f"RankSampler({props})"


class ProportionalSampler:
    """ Implements the proportional-based sampling in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf).
    """

    # pylint: disable=too-many-instance-attributes
    # Nine is reasonable in this case.
    def __init__(self, capacity, batch_size=1, collate=None, **kwargs):
        self.__data = [None for _ in range(capacity)]
        self.__sumtree = SumTree(capacity=capacity)
        self.__capacity = capacity
        self.__batch_size = batch_size
        self.__collate = collate or _collate_with_index
        self.__alpha = kwargs["alpha"] if "alpha" in kwargs else 0.9
        self.__epsilon = (
            kwargs["epsilon"] if "epsilon" in kwargs else 0.000_000_1
        )
        self.__pos = 0

    def push(self, transition):
        """ Push new transition to the experience replay. If priority not
        available then initialize with a large priority making sure every new
        transition is being sampled and updated.
        """
        priority = self.__epsilon ** self.__alpha + 1000
        self.__sumtree.update(self.__pos, priority)
        self.__data[self.__pos] = transition
        self.__pos = (self.__pos + 1) % self.__capacity

    def update(self, idx, priority):
        """ Updates the priority of a given transition. """
        priority = (priority + self.__epsilon) ** self.__alpha
        self.__sumtree.update(idx, priority)

    def sample(self):
        # TODO: docstring

        segment_sz = self.__sumtree.get_sum() / self.__batch_size
        samples = []
        for i in range(self.__batch_size):
            a = i * segment_sz
            b = (i + 1) * segment_sz
            s = np.random.uniform(a, b)
            idx, _ = self.__sumtree.get(s)
            samples.append((idx, self.__data[idx]))

        return self.__collate(samples)

    def __len__(self):
        return len(self.__data)

    def __str__(self):
        props = f"size={len(self)}, α={self.__alpha}, batch={self.__batch_size}"
        return f"ProportionalSampler({props})"


class BootstrappedER:
    """ Wrapper over any ER implementation providing also a mechanism for data
        bootstrapping.
    """

    def __init__(self, sampling, er_args, boot_p, boot_no, shuffle):
        self.__delegate = BUFFERS[sampling](**er_args)
        self.__boot_dist = torch.distributions.Bernoulli(boot_p)
        self.__boot_no = boot_no
        self.__shuffle = shuffle

    def sample_mask(self):
        """ Returns a binary tensor that masks the data for an ensemble.
        """
        return self.__boot_dist.sample((self.__boot_no,))

    @property
    def shuffle(self):
        return self.__shuffle

    def __getattr__(self, name):
        return getattr(self.__delegate, name)

    def __str__(self):
        return f"Bootsrapped{str(self.__delegate)}"

    def __len__(self):
        return len(self.__delegate)


BUFFERS = {
    "uniform": NaiveExperienceReplay,
    "greedy-pq": GreedyPQSampler,
    "greedy-hpq": GreedyHeapqSampler,
    "rank": RankSampler,
    "proportional": ProportionalSampler,
}
