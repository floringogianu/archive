""" Some general utils. """
import os
from time import time
from argparse import Namespace

import torch


def get_ground_truth(n, gamma):
    """ Compute the ground truth values for the BlindCliffWalk MDP. """
    test_states = torch.cat([torch.ones(n, 1), torch.eye(n)], 1)
    ground_truth = torch.zeros(n, 2)
    good_actions_values = [gamma ** i for i in range(n)]

    for i, q in enumerate(reversed(good_actions_values)):
        act_idx = 0 if i % 2 == 0 else 1
        ground_truth[i][act_idx] = q

    return test_states, ground_truth


def create_paths(args: Namespace) -> Namespace:
    if not hasattr(args, "out_dir") or args.out_dir is None:
        if not os.path.isdir("./results"):
            os.mkdir("./results")
        out_dir = f"./results/{str(int(time())):s}_{args.experiment:s}"
        os.mkdir(out_dir)
        args.out_dir = out_dir
    elif not os.path.isdir(args.out_dir):
        raise Exception(f"Directory {args.out_dir} does not exist.")

    if not hasattr(args, "run_id"):
        args.run_id = 0

    return args
