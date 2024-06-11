import os
import random

import numpy as np
import torch


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sampler(B, trajectories, max_batch_num):
    permutation = torch.randperm(trajectories[0].size(0))
    batch_idxs = torch.split(permutation, B)  # list of batches of transition idxs
    batch_idxs = batch_idxs[: (max_batch_num or len(batch_idxs))]
    for idxs in batch_idxs:
        yield [el[idxs, :] for el in trajectories]
