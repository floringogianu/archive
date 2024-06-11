import os
import random

import numpy as np
import torch

from src.util.io_util import (
    checkpoint_agent,
    config_to_string,
    dict_to_namespace,
    namespace_to_dict,
    save_config,
    read_config
)
from src.util.random_search_util import maybe_sample_hyperparams_


def seed_everything(seed):
    """Set the seed on everything I can think of.
    Hopefully this should ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


__all__ = [
    "config_to_string",
    "save_config",
    "read_config",
    "dict_to_namespace",
    "namespace_to_dict",
    "checkpoint_agent",
    "maybe_sample_hyperparams_",
    "seed_everything",
]
