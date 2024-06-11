import math
import random

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

from src import get_datasets, get_model
from src.io_utils import read_config, config_to_string
from train_dlv3 import accumulate_gradients


def get_options(cmdl):
    opt = read_config(cmdl.cfg)  # read config file and augment settings
    opt.verbose = cmdl.verbose
    opt.extra_name = cmdl.name
    return opt


def find_initial_lr(
    model, loader, optimizer, scheduler, device, max_step_no, beta=0.98
):

    avg_loss, best_loss = 0, 0
    losses, lrs = [], []

    loader_iterator = iter(loader)
    for i in range(max_step_no):

        # fetch data
        try:
            images, targets = next(loader_iterator)
        except StopIteration:
            # end of the rope
            loader_iterator = iter(loader)
            images, targets = next(loader_iterator)

        # cuda
        images = images.to(device)
        targets = targets.to(device)

        loss = accumulate_gradients(1, model, images, targets)

        # record loss
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))
        lr = scheduler.get_lr()

        # Stop if the loss is exploding
        if i > 0 and smoothed_loss > 4 * best_loss:
            return lrs, losses

        # Record the best loss
        if smoothed_loss < best_loss or i == 0:
            best_loss = smoothed_loss

        print(
            "{:04d}/{:04d} loss={:2.3f} / {:2.3f}, lr={:2.8f}".format(
                i, max_step_no, smoothed_loss, best_loss, lr[0]
            )
        )

        if i == max_step_no:
            return lrs, losses

        # Store the values
        losses.append(smoothed_loss)
        lrs.append(lr[0])

        # Optimize and step
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return lrs, losses


def main(opt):
    # Set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = opt.cudnn_benchmark  # activate perf-optim

    device = torch.device(opt.device)

    # Configure datasets
    train_set, _ = get_datasets(opt)
    train_loader = torch.utils.data.DataLoader(train_set, **vars(opt.loader))

    # Configure model
    model = get_model(opt)
    model.to(device)
    model = torch.nn.DataParallel(model)

    # Configure LR range
    lr_start = 1e-6
    lr_end = 0.9
    max_step_no = 2 * (len(train_loader) - 1)
    opt.optim.args.lr = lr_start

    # Optimizer
    optimizer = SGD(params=model.parameters(), **opt.optim.args.__dict__)

    # Configure scheduler
    gamma = (lr_end / lr_start) ** (1 / max_step_no)
    scheduler = ExponentialLR(optimizer, gamma, last_epoch=-1)

    print(config_to_string(opt), "\n")
    print(train_set)
    print("Batches:    ", len(train_loader))
    print("Iterations: ", max_step_no)
    print("Start={:2.8f}, End={:2.8f}".format(lr_start, lr_end))

    lrs, losses = find_initial_lr(
        model,
        train_loader,
        optimizer,
        scheduler,
        device,
        max_step_no,
        beta=0.98,
    )

    fp = "./results/findLR_{:s}_i-{:d}_s-{:s}_r-{:1.1f}_b-{:2d}".format(
        opt.dataset.name,
        max_step_no,
        str(opt.dataset.styled),
        opt.dataset.style_ratio,
        opt.loader.batch_size,
    )
    if opt.extra_name:
        fp += f"_{opt.extra_name}"
    fp += ".pth"

    torch.save(
        {
            "lrs": lrs,
            "losses": losses,
            "meta": {
                "dset": opt.dataset.name,
                "steps": len(lrs),
                "max_steps_no": max_step_no,
                "styled": opt.dataset.styled,
                "ratio": opt.dataset.style_ratio,
                "batch_size": opt.loader.batch_size,
                "extra_name": opt.extra_name,
            },
        },
        fp,
    )
    print("Saved to: ", fp)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(description="ShapeBias")
    PARSER.add_argument(
        "--cfg",
        "-c",
        type=str,
        default="./configs/vkitti_deeplab_v3.yml",
        help="Path to the configuration file."
        + "Default is `./configs/vkitti_deeplab_v3.yml`",
    )
    PARSER.add_argument(
        "--name", "-n", default=None, type=str, help="Extra name."
    )
    PARSER.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="If the logging should be more verbose.",
    )
    main(get_options(PARSER.parse_args()))
