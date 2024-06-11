""" Training protocol as described in https://arxiv.org/abs/1706.05587
"""
import gc
import random
from pathlib import Path

import neptune
import numpy as np
import rlog
import torch
import torch.nn.functional as F
import yaml

from src import get_datasets, get_model, get_optimizer
from src.datasets import resize_labels
from src.io_utils import (
    config_to_string,
    create_paths,
    flatten_namespace,
    get_git_info,
    namespace_to_dict,
    read_config,
)
from src.utils import get_ious, reset_optimizer_


def accumulate_gradients(opt, model, images, targets):
    """ Accumulates gradients and returns loss and iou stats.

        The graph is cleared every time this function is called.
    """

    acc_loss, acc_ious = 0, []
    for _ in range(opt.grad_acc_steps):

        # inference
        if opt.model.model in ("deeplabv2_resnet101", "deeplabv2_vgg16"):
            logits = model(images)

            loss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, height, width = logit.shape
                labels_ = resize_labels(targets, size=(height, width))
                loss += F.cross_entropy(logit, labels_.to(logit.device))

            _, height, width = targets.shape
            outputs = F.interpolate(
                logits[-1],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            out_idxs = torch.argmax(outputs, dim=1)

        else:
            targets = targets.to(images.device)
            outputs = model(images)["out"]
            out_idxs = outputs.argmax(1)
            loss = F.cross_entropy(outputs, targets)

        # accumulate gradients, clear the graph
        loss.backward()

        # compute ious per example
        for prediction, target in zip(out_idxs.data, targets.data):
            acc_ious.append(
                get_ious(prediction.unsqueeze(0), target.unsqueeze(0))[0]
            )

        # accumulate stats
        acc_loss += loss.data.item()

    return acc_loss / opt.grad_acc_steps, torch.tensor(acc_ious).mean().item()


def set_bn_eval(module):
    """ Put all BatchNorm layers in eval mode.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def train(opt, step_range, loader, model, optim, device, scheduler=None):
    """ Training routine.
    """
    train_log = rlog.getLogger(opt.experiment + ".train")
    train_log.info("Starting training range.")
    train_log.info("[optim steps/total_steps] other metrics")
    train_log.info("Initial lr=%.8f.", optim.param_groups[0]["lr"])
    log_fmt = (
        "[{0:05d}/{1:05d}][{2:05d}] "
        + "avg_loss={avg_loss:2.3f}, avg_iou={avg_iou:2.2f}  |  lr={lr:.8f}."
    )

    model.train()
    # freeze bnn_layers
    if opt.freeze_bnn:
        rlog.info("Freezing BatchNorm layers during training.")
        model.module.classifier.apply(set_bn_eval)

    loader_iterator = iter(loader)

    step_cnt = None  # we use this after the loop
    for step_cnt in step_range:

        # restart optimizer
        if hasattr(opt, "cyclic"):
            optim, scheduler = reset_optimizer_(
                opt, step_cnt, model, optim, scheduler
            )

            # TODO: this needs to be called only when the scheduler changes
            # eats a lot of time
            gc.collect()
            torch.cuda.empty_cache()

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
        loss, iou = accumulate_gradients(opt, model, images, targets)

        # optimization
        optim.step()
        optim.zero_grad()
        if scheduler is not None:
            scheduler.step()

        # stats
        train_log.put(loss=loss, iou=iou)
        try:
            neptune.send_metric("train_loss", loss)
            neptune.send_metric("train_iou", iou)
        except neptune.exceptions.Uninitialized as err:
            rlog.debug("Nepune send_metric exception. Run with verbose.")
            if opt.verbose:
                rlog.exception(err)

        if step_cnt % opt.log_freq == 0:
            summary = train_log.summarize()
            train_log.info(
                log_fmt.format(
                    step_cnt,
                    opt.step_no,
                    step_cnt * loader.batch_size,
                    **summary,
                    lr=optim.param_groups[0]["lr"],
                )
            )
            train_log.trace(step=step_cnt, **summary)
            train_log.reset()

    try:
        # if logger did not just reset, log the final batch
        summary = train_log.summarize()
        train_log.info(log_fmt.format(step_cnt, step_no, **summary))
        train_log.trace(step=step_cnt, **summary)
        train_log.reset()
    except:  # pylint:disable=bare-except
        pass

    train_log.info("Saving model...")
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    torch.save(
        {
            "step": step_cnt,
            # "epoch": summary["epoch"],
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler_state,
            "avg_loss10": summary["avg_loss"],
            "avg_iou10": summary["avg_iou"],
        },
        f"{opt.out_dir}/model_e{step_cnt:02d}.pth",
    )
    return optim, scheduler


def validate(opt, step_cnt, loader, model, device):
    """ Validation routine.
    """
    val_log = rlog.getLogger(opt.experiment + ".val")
    val_log.info("Starting validation.")

    model.eval()
    for batch_cnt, (images, targets) in enumerate(loader):
        with torch.no_grad():
            # cuda
            images = images.to(device)
            targets = targets.to(device)

            # inference
            if opt.model.model in ("deeplabv2_resnet101", "deeplabv2_vgg16"):
                outputs = model(images)
                _, height, width = targets.shape

                outputs = F.interpolate(
                    outputs,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                outputs = model(images)["out"]
            out_idxs = outputs.argmax(1)

            loss = F.cross_entropy(outputs, targets)
            # compute ious per example
            ious = []
            for prediction, target in zip(out_idxs.data, targets.data):
                ious.append(
                    get_ious(prediction.unsqueeze(0), target.unsqueeze(0))[0]
                )
            iou = torch.tensor(ious).mean().item()

        val_log.put(loss=loss.data.item(), iou=iou)
        if batch_cnt % 5 == 0 and batch_cnt != 0:
            print(".", end="")

    log_fmt = (
        "\nValidation @{:6d}: avg_loss={avg_loss:2.3f}, avg_iou={avg_iou:2.2f}."
    )
    summary = val_log.summarize()
    val_log.info(log_fmt.format(step_cnt, **summary))
    val_log.trace(step=step_cnt, **summary)
    val_log.reset()
    try:
        neptune.send_metric("val_loss", summary["avg_loss"])
        neptune.send_metric("val_iou", summary["avg_iou"])
    except neptune.exceptions.NoExperimentContext as err:
        rlog.debug("Nepune send_metric exception. Run with verbose.")
        if opt.verbose:
            rlog.exception(err)


def maybe_resume_(cmdl, opt):
    """ Check if an experiment is being resumed and set in options.
    """
    #
    if cmdl.resume is None:
        opt.resumed = False
    else:
        chkpt = Path(cmdl.resume)
        assert chkpt.is_file(), f"{str(chkpt)} is not the path to a checkpoint."
        opt.resumed = True
        opt.checkpoint_path = cmdl.resume
        opt.out_dir = chkpt.parent


def get_options(cmdl):
    opt = read_config(cmdl.cfg)  # read config file and augment settings
    opt.verbose = cmdl.verbose
    maybe_resume_(cmdl.resume, opt)
    return opt


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
    rlog.init(opt.experiment, opt.out_dir)
    train_log = rlog.getLogger(opt.experiment + ".train")
    train_log.addMetrics(
        [
            # rlog.SumMetric("epoch", resetable=False, metargs=["epoch"]),
            rlog.ValueMetric("loss", metargs=["loss"]),
            rlog.AvgMetric("avg_loss", metargs=["loss", 1]),
            rlog.ValueMetric("iou", metargs=["iou"]),
            rlog.AvgMetric("avg_iou", metargs=["iou", 1]),
            rlog.MaxMetric("max_iou", metargs=["iou"]),
        ]
    )
    val_log = rlog.getLogger(opt.experiment + ".val")
    val_log.addMetrics(
        [
            rlog.AvgMetric("avg_loss", metargs=["loss", 1]),
            rlog.AvgMetric("avg_iou", metargs=["iou", 1]),
        ]
    )

    # Configure datasets
    train_set, val_set = get_datasets(opt)

    train_log.info("Train set: %s ", train_set)
    val_log.info("Validation set: %s", val_set)
    train_log.info("Joint transform: %s", train_set._joint_transforms)
    val_log.info("Joint transform: %s", val_set._joint_transforms)

    # Configure model
    model = get_model(opt)
    model.to(device)
    model = torch.nn.DataParallel(model)

    if opt.verbose:
        rlog.info(model)

    # Optimizer
    optimizer, scheduler = get_optimizer(model, opt)

    # Load object states required for resuming training.
    step_cnt, epoch_cnt = 1, 0
    if opt.resumed is not None:
        rlog.info(f"Resuming {opt.experiment} experiment.")
        chkpt = torch.load(chkpt)
        epoch_cnt = chkpt["epoch"]
        step_cnt = chkpt["step"]
        model.load_state_dict(chkpt["model_state"])
        try:
            optimizer.load_state_dict(chkpt["optim_state"])
            scheduler.load_state_dict(chkpt["scheduler_state"])
        except KeyError as err:
            rlog.warning("Optim/scheduler states not in the checkpoint.")
        rlog.info(f"Resuming experiment from {Path(opt.checkpoint_path)}...")
        rlog.info(f"Epoch {epoch_cnt}, step {step_cnt}.")
        # cleanup
        del chkpt
        gc.collect()
        torch.cuda.empty_cache()
    else:
        rlog.info(f"Configured {opt.experiment} experiment.")

    # And configure Neptune
    try:
        if opt.resume is not None:
            session = neptune.sessions.Session()
            project = session.get_project("shape-bias/segmentation")
            with open(Path(opt.out_dir) / "neptune_id") as file:
                neptune_id = file.readline()
            neptune_experiment = project.get_experiments(id=neptune_id)[0]
            rlog.info(f"Neptune experiment {neptune_id} resuming.")
        else:
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

    # Log the configuration
    rlog.info(f"Experiment configuration:\n{config_to_string(opt)}")
    yml_path = f"{opt.out_dir}/cfg.yml"
    rlog.info(f"Saving config in experiment folder: {yml_path}")
    with open(yml_path, "w") as file:
        yaml.safe_dump(namespace_to_dict(opt), file, default_flow_style=False)

    # Start training  ----------------------------------------------------------
    train_rounds = [
        (i * opt.val_freq + 1, (i * opt.val_freq + 1) + opt.val_freq)
        for i in range(opt.step_no // opt.val_freq)
    ]  # [(1, 5001), (5001, 10001), etc]
    try:

        for start_step, end_step in train_rounds:
            optimizer, scheduler = train(
                opt,
                range(start_step, end_step),
                torch.utils.data.DataLoader(train_set, **vars(opt.loader)),
                model,
                optimizer,
                device,
                scheduler=scheduler,
            )
            # cleanup
            gc.collect()
            torch.cuda.empty_cache()

            # validate
            validate(
                opt,
                end_step,
                torch.utils.data.DataLoader(val_set, batch_size=24),
                model,
                device,
            )
            # cleanup
            gc.collect()
            torch.cuda.empty_cache()

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
