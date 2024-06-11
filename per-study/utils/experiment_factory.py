""" Functions and classes related to experiment configuration.
"""
from functools import partial

from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
import torch.distributions as D
import gym
from utils import get_all_transitions
from experience_replay import get_experience_replay
from bootstrapp import BootstrappedEstimator


LOSS_F = {"mse": F.mse_loss, "huber": F.smooth_l1_loss}


class StochasticRewards:
    """ A wrapper over experience replay implementations that adds noise with
    a given precision to the rewards. We use this to simulate a stochastic
    environment within our experimental setup which has a fixed size buffer
    containing the transitions corresponding to an uniform exploration.
    """

    def __init__(self, delegate, noise_precision):
        """ StochasticRewards constructor.
        Args:
            delegate (ExperienceReplay): The ER buffer we sample from.
            noise_precision (float): The noise precision.
        """
        self.__delegate = delegate
        self.__noise_precision = noise_precision
        self.__pos = D.Normal(1, noise_precision)
        self.__neg = D.Normal(0, noise_precision)

    def sample(self):
        """Adds noise to the reward in the sampled transitions.

        Returns:
            list: A batch of transitions.
        """
        pos, neg = self.__pos, self.__neg
        batch = self.__delegate.sample()
        batch_ = []
        for transition in batch:
            s, a, r, s_, d, m = transition
            r = neg.sample(r.shape) if m["win"] == 0 else pos.sample(r.shape)
            batch_.append((s, a, r, s_, d, m))
        return batch_

    def __getattr__(self, name):
        return getattr(self.__delegate, name)

    def __str__(self):
        beta = self.__noise_precision
        return f"StochasticRewards({str(self.__delegate)}, β={beta})."

    def __len__(self):
        return len(self.__delegate)


def configure_experiment(opt, lr, learners):
    """ Sets up the objects required for running the experiment.
    """
    # set some variables
    n = opt.mdp_size
    gamma = 1 - 1 / n
    learn, learn_ensemble = learners
    try:
        bayesian = opt.experience_replay.bayesian
        boot_no = opt.experience_replay.boot_no
        boot_beta = opt.experience_replay.boot_beta
    except AttributeError:
        bayesian = False

    # sample the env
    env = gym.make(f"BlindCliffWalk-N{n}-v0")
    transitions = get_all_transitions(env, n)

    # construct and populate Experience Replay
    mem, cb = get_experience_replay(
        len(transitions), **opt.experience_replay.__dict__
    )
    if bayesian:
        for transition in transitions:
            meta = {"boot_mask": mem.sample_mask(), "win": transition[2]}
            mem.push((*transition, meta))
    else:
        for transition in transitions:
            # we need this meta information for the experiments with stochastic
            # rewards
            meta = {"win": transition[2]}
            mem.push((*transition, meta))

    # wrap the experience replay in a class that adds noise to the rewards
    # returned every time we call `mem.sample()`.
    try:
        if opt.reward_noise_precision:
            mem = StochasticRewards(mem, opt.reward_noise_precision)
    except AttributeError:
        pass

    # loss function
    try:
        loss_fn = LOSS_F[opt.loss]
    except AttributeError:
        loss_fn = LOSS_F["mse"]

    # configure estimator and optimizer
    estimator = nn.Linear(n + 1, 2, bias=False)  # already added in the state
    estimator.weight.data.normal_(0, 0.1)
    if bayesian:
        estimator = BootstrappedEstimator(estimator, B=boot_no, beta=boot_beta)

    optimizer = SGD(estimator.parameters(), lr=lr)
    optimizer.zero_grad()

    # get sampling type tag, we use it for reporting
    try:
        tag = opt.experience_replay.__name
        rw_noise = opt.reward_noise_precision
        tag = f"{tag}_det" if rw_noise == 0 else f"{tag}_stoch:{rw_noise}"
    except AttributeError:
        tag = get_sampling_variant(**opt.experience_replay.__dict__)

    # add loss name in tag if it exists
    tag = f"{tag}_{opt.loss}" if "loss" in opt.__dict__ else tag

    # configure learner
    learner = partial(learn, gamma=gamma, loss_fn=loss_fn, optimizer=optimizer)
    if bayesian:
        learner = partial(learn_ensemble, learner=learner, mem=mem)

    print(f">>  Experience Replay: {mem}")
    return mem, cb, estimator, learner, tag


def get_sampling_variant(sampling="uniform", **kwargs):
    """ Creates a tag of the form sampling + hyperparams if hyperparams exist
    """
    hp_names = (
        "alpha",
        "beta",
        "batch_size",
        "bayesian",
        "boot_shuffle",
        "boot_p",
        "boot_beta",
    )
    hyperparams = {h: kwargs[h] for h in hp_names if h in kwargs}

    if not kwargs:
        return sampling
    for k, v in hyperparams.items():
        v = int(v) if isinstance(v, bool) else v
        sampling += f"_{k.replace('boot_', 'b')}:{v}"
    return sampling
