""" Training protocol as described in https://arxiv.org/abs/1706.05587
"""
import gc
import random
from functools import partial
from pathlib import Path

import neptune
import numpy as np
import rlog
import torch
import torch.nn.functional as F
import yaml

from src import get_datasets, get_model, get_optimizer
from src.datasets import KITTI2VKITTI, CITYSCAPES2SYNTHIA, remap_target
from src.io_utils import (
    config_to_string,
    create_paths,
    flatten_namespace,
    get_git_info,
    namespace_to_dict,
    read_config,
)
from src.utils import get_ious, wtf_mem

CLASSES = {
    "VKITTI": range(14),
    "KITTI": list(KITTI2VKITTI.values()),
    "SYNTHIA": range(23),
    "Cityscapes": list(CITYSCAPES2SYNTHIA.values()),
}


def accumulate_gradients(grad_acc_steps, model, images, targets):
    """ Accumulates gradients and returns loss and iou stats.
        The graph is cleared every time this function is called.
    """

    acc_loss = 0
    for _ in range(grad_acc_steps):

        # inference
        targets = targets.to(images.device)
        outputs = model(images)["out"]
        loss = F.cross_entropy(outputs, targets)

        # accumulate gradients, clear the graph
        loss.backward()

        # accumulate stats
        acc_loss += loss.data.item()

    return acc_loss / grad_acc_steps


def do_stats(step_cnt, loss, lr, opt):
    """ Hide the mess in a function. Handles calls to rlog and neptune.
    """
    log = rlog.getLogger(f"{opt.experiment}.train")
    log_fmt = "[{0:05d}/{1:05d}] avg_loss={avg_loss:2.3f}  |  lr={lr:.8f}."

    # stats
    log.put(loss=loss)
    try:
        neptune.send_metric("train_loss", loss)
        neptune.send_metric("lr", lr)
    except neptune.exceptions.Uninitialized as err:
        rlog.debug("Nepune send_metric exception. Run with verbose.")
        if opt.verbose:
            rlog.exception(err)

    if step_cnt % opt.log_freq == 0:
        summary = log.summarize()
        log.info(log_fmt.format(step_cnt, opt.step_no, **summary, lr=lr))
        log.trace(step=step_cnt, **summary)
        log.reset()


def train(opt, step_range, loader, model, optim, device, scheduler=None):
    """ Training routine.
    """
    log = rlog.getLogger(opt.experiment + ".train")
    log.info("Starting training range.")
    log.info("Initial lr=%.8f.", optim.param_groups[0]["lr"])
    log.reset()

    model.train()
    step_cnt = None  # we use this after the loop
    loader_iterator = iter(loader)

    for step_cnt in step_range:

        # fetch data
        try:
            images, targets = next(loader_iterator)
        except StopIteration:
            # end of the rope
            loader_iterator = iter(loader)
            images, targets = next(loader_iterator)

        # cuda
        images = images.to(device)

        # accumulate gradients
        loss = accumulate_gradients(opt.grad_acc_steps, model, images, targets)

        # optimization
        optim.step()
        optim.zero_grad()

        # increment scheduler
        if scheduler is not None:
            scheduler.step()

        # log stuff
        do_stats(step_cnt, loss, scheduler.get_lr()[0], opt)

    # training range ends, do one more logging if required.
    try:
        # if logger did not just reset, log the final batch
        summary = log.summarize()
        log.info(
            "[{0:05d}/{1:05d}] avg_loss={avg_loss:2.3f}  |  lr={lr:.8f}.".format(
                step_cnt, opt.step_no, **summary, lr=scheduler.get_lr()
            )
        )
        log.trace(step=step_cnt, **summary)
        log.reset()
    except ZeroDivisionError:
        log.info("All good...")

    del loader_iterator
    del loader


def validate(opt, step_cnt, loader, model, labels, device, remap_fn=None):
    """ Validation routine.
    """
    val_name = "source_val" if remap_fn is None else "target_val"
    log = rlog.getLogger(f"{opt.experiment}.{val_name}")
    log.info("Starting %s validation.", val_name)
    log.reset()

    model.eval()
    for batch_cnt, (images, targets) in enumerate(loader):
        with torch.no_grad():
            # cuda
            images = images.to(device)
            targets = targets.long().to(device)

            if remap_fn is not None:
                targets = remap_fn(targets)

            # inference
            outputs = model(images)["out"]
            out_idxs = outputs.argmax(1)

            # compute loss (don't know how just yet)
            # TODO: fix the class discrepancy between datasets
            # losses.append(F.cross_entropy(outputs, targets)
            # losses.append(0.42)
            if remap_fn is None:
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = torch.tensor(0.42)  # pylint: disable=E1102

            # compute mIoU per example
            ious = []
            for prediction, target in zip(out_idxs.data, targets.data):
                ious.append(
                    get_ious(
                        prediction.unsqueeze(0),
                        target.unsqueeze(0),
                        labels=labels,
                    )[0]
                )
            iou = torch.tensor(ious).mean().item()  # pylint: disable=E1102

        log.put(loss=loss.data.item(), iou=iou)
        if batch_cnt % 5 == 0 and batch_cnt != 0:
            print(".", end="")

    log_fmt = (
        "\nValidation @{:6d}: avg_loss={avg_loss:2.3f}, avg_iou={avg_iou:2.2f}."
    )
    summary = log.summarize()
    log.info(log_fmt.format(step_cnt, **summary))
    log.trace(step=step_cnt, **summary)
    log.reset()
    try:
        neptune.send_metric(f"{val_name}_loss", summary["avg_loss"])
        neptune.send_metric(f"{val_name}_iou", summary["avg_iou"])
    except neptune.exceptions.Uninitialized as err:
        rlog.debug("Nepune send_metric exception. Run with verbose.")
        if opt.verbose:
            rlog.exception(err)

    return summary["avg_loss"], summary["avg_iou"]


def maybe_resume_(checkpoint_path, opt):
    """ Check if an experiment is being resumed and set in options.
    """
    if checkpoint_path is None:
        opt.resumed = False
    else:
        chkpt = Path(checkpoint_path)
        assert chkpt.is_file(), f"{str(chkpt)} is not the path to a checkpoint."
        opt.resumed = True
        opt.checkpoint_path = checkpoint_path
        opt.out_dir = chkpt.parent


def get_options(cmdl):
    """ Augments the options namespace.
    """
    opt = read_config(cmdl.cfg)  # read config file and augment settings
    opt.verbose = cmdl.verbose
    maybe_resume_(cmdl.resume, opt)
    return opt


def configure_logger(opt, with_target_dset=False):
    """ Configure rLog. """
    rlog.init(opt.experiment, opt.out_dir)
    train_log = rlog.getLogger(opt.experiment + ".train")
    train_log.addMetrics(
        [
            # rlog.SumMetric("epoch", resetable=False, metargs=["epoch"]),
            rlog.ValueMetric("loss", metargs=["loss"]),
            rlog.AvgMetric("avg_loss", metargs=["loss", 1]),
        ]
    )
    src_val_log = rlog.getLogger(opt.experiment + ".source_val")
    src_val_log.addMetrics(
        [
            rlog.AvgMetric("avg_loss", metargs=["loss", 1]),
            rlog.AvgMetric("avg_iou", metargs=["iou", 1]),
        ]
    )
    if with_target_dset:
        tgt_val_log = rlog.getLogger(opt.experiment + ".target_val")
        tgt_val_log.addMetrics(
            [
                rlog.AvgMetric("avg_loss", metargs=["loss", 1]),
                rlog.AvgMetric("avg_iou", metargs=["iou", 1]),
            ]
        )
        return train_log, src_val_log, tgt_val_log
    return train_log, src_val_log


def configure_neptune(opt):
    """ Configure Neptune. """
    try:
        if opt.resumed:
            raise ValueError("Resuming Neptune experiments is not implemented.")
        neptune_experiment = None
        neptune.init("shape-bias/segmentation")
        neptune_experiment = neptune.create_experiment(
            name=opt.experiment, upload_source_files=["train.py", "src"]
        )
        for key, value in flatten_namespace(opt).items():
            neptune.set_property(key, str(value))
        with open(Path(opt.out_dir) / "neptune_id", "w") as file:
            neptune_id = neptune_experiment.id
            file.write(neptune_id)
        rlog.info(f"Neptune experiment {neptune_id} enabled.")
        opt.neptune_id = neptune_experiment.id
    except neptune.exceptions.NeptuneException as err:
        if opt.verbose:
            rlog.exception(err)
        rlog.warning("Neptune disabled")
    return neptune_experiment


def maybe_pretrained(model, opt):
    if hasattr(opt, "pretrained"):
        chkpt_path = opt.pretrained.checkpoint
        rlog.info("Loading model from %s", chkpt_path)
        chkpt = torch.load(chkpt_path)
        model.load_state_dict(chkpt["model_state"])
    return model


def save_checkpoint(out_dir, crt_step, model, optimizer, scheduler, stats):
    """ Save the state of the experiment. """
    rlog.info("Saving model...")
    torch.save(
        {
            "step": crt_step,
            # "epoch": summary["epoch"],
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "stats": stats,
        },
        f"{out_dir}/model_e{crt_step:06d}.pth",
    )


def main(opt):
    """ Reads config file, constructs objects and calls training and validation
        routines.
    """

    # Set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = opt.cudnn_benchmark  # activate perf-optim

    device = torch.device(opt.device)

    opt.git = get_git_info()
    if not opt.resumed:
        create_paths(opt)  # create experiment folders and set paths

    # Configure logger
    train_log, src_val_log, tgt_val_log = configure_logger(opt, True)

    # Configure datasets
    train_set, val_set, target_set = get_datasets(opt, with_target_dset=True)

    train_log.info("Train set: %s ", train_set)
    src_val_log.info("Source validation set: %s", val_set)
    tgt_val_log.info("Target validation set: %s", target_set)
    # pylint: disable=protected-access
    train_log.info("Joint transform: %s", train_set._joint_transforms)
    src_val_log.info("Joint transform: %s", val_set._joint_transforms)
    tgt_val_log.info("Joint transform: %s", target_set._joint_transforms)
    # pylint: enable=protected-access

    # Configure model
    model = get_model(opt)
    model.to(device)

    # maybe pretrain
    model = maybe_pretrained(model, opt)

    if opt.verbose:
        rlog.info(model)

    # Optimizer
    optimizer, scheduler = get_optimizer(model, opt)

    # Load object states required for resuming training.
    if opt.resumed:
        raise ValueError("Resuming experiments is not implemented.")
    rlog.info(f"Configured {opt.experiment} experiment.")

    # And configure Neptune
    neptune_experiment = configure_neptune(opt)

    # Log the configuration
    rlog.info(f"Experiment configuration:\n{config_to_string(opt)}")

    # And save it in the experiment folder
    with open(f"{opt.out_dir}/cfg.yml", "w") as file:
        yaml.safe_dump(namespace_to_dict(opt), file, default_flow_style=False)

    # Start training, finally!
    vlf, sno = opt.val_freq, opt.step_no
    train_rounds = [
        (i * vlf + 1, (i * vlf + 1) + vlf) for i in range(sno // vlf)
    ]  # [(1, 5001), (5001, 10001), etc]

    try:
        for start_step, end_step in train_rounds:
            train(
                opt,
                range(start_step, end_step),
                torch.utils.data.DataLoader(train_set, **vars(opt.loader)),
                model,
                optimizer,
                device,
                scheduler=scheduler,
            )

            # validate
            val_loss, val_iou = validate(
                opt,
                end_step,
                torch.utils.data.DataLoader(val_set, batch_size=24),
                model,
                CLASSES["VKITTI"],
                device,
            )

            # validate on target
            if target_set is not None:
                mapping_dict = (
                    KITTI2VKITTI
                    if opt.dataset.name == "VKITTI"
                    else CITYSCAPES2SYNTHIA
                )
                tgt_val_log.info(
                    "\nTargets will be remapped as follows:\n%s.",
                    str(mapping_dict),
                )
                validate(
                    opt,
                    end_step,
                    torch.utils.data.DataLoader(target_set, batch_size=24),
                    model,
                    CLASSES["KITTI"],
                    device,
                    remap_fn=partial(remap_target, mapping=mapping_dict),
                )

            # save model
            val_loss, val_iou = 0, 0
            save_checkpoint(
                opt.out_dir,
                end_step - 1,
                model,
                optimizer,
                scheduler,
                {"val_loss": val_loss, "val_iou": val_iou},
            )

            gc.collect()

            rlog.info(wtf_mem(topk=10))

    except Exception as err:
        if neptune_experiment:
            neptune_experiment.stop(exc_tb=str(err))
        raise err
    else:
        if neptune_experiment:
            neptune_experiment.stop(exc_tb=None)


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
        "--resume",
        "-r",
        type=str,
        help="The path to a checkpoint you want to resume.",
    )
    PARSER.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="If the logging should be more verbose.",
    )
    main(get_options(PARSER.parse_args()))
