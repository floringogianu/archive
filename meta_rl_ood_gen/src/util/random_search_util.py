""" Utils for sampling hyperparameters from an
ill-defined specification in a config file.
"""
from argparse import Namespace
from types import NoneType

import numpy as np


def _maybe_number(something):
    if isinstance(something, (bool, NoneType)):
        return something
    try:
        number = float(something)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        return str(something)


def logUniform(low, high):
    """Sample from exp(U( log(a), log(b) ))"""
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


SAMPLING_FNS = {
    "uniform": lambda sup: float(np.random.uniform(*sup)),
    "uniformInt": lambda sup: int(np.random.randint(*sup)),
    "logUniform": lambda sup: float(logUniform(*sup)),
    "logUniformInt": lambda sup: int(logUniform(*sup)),
    "choice": lambda sup: _maybe_number(np.random.choice(sup)),
}


def maybe_sample_hyperparams_(opt):
    """Finds values in the configuration Namespace of the form [sampler, [*domain]]
    and replaces it with a random sample accordingly.
    """
    iterable = opt.__dict__.items() if isinstance(opt, Namespace) else opt.items()
    for key, value in iterable:
        if isinstance(value, (Namespace, dict)):
            maybe_sample_hyperparams_(value)
        elif isinstance(value, list) and value[0] in SAMPLING_FNS:
            sampler, support = value
            support = [_maybe_number(x) for x in support]
            # sometimes the support is a list of strings
            try:
                support = list(sorted([_maybe_number(x) for x in support]))
            except TypeError:
                pass
            sample = SAMPLING_FNS[sampler](support)
            # opt can be either a Namespace or a dict at this point
            try:
                setattr(opt, key, sample)
            except AttributeError:
                opt[key] = sample
        else:
            pass
