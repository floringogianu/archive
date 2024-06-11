""" Evaluate on the real counterparts.
"""
import argparse
import gc
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import get_datasets, get_model
from src.datasets import (
    KITTI2VKITTI,
    CITYSCAPES2SYNTHIA,
    remap_target,
    VirtualKITTI,
    SYNTHIA,
)
from src.io_utils import YamlNamespace, config_to_string, read_config
from src.utils import StatsCollector, IoUMetric, ConfusionMatrix

NAME2PATH = {
    "VKITTI": "./data/vkitti/",
    "KITTI": "./data/rkitti/",
    "SYNTHIA": "./data/SYNTHIA_RAND_CITYSCAPES/",
    "Cityscapes": "./data/Cityscapes-HalfSize",
}
SYNTH2REAL = {"VKITTI": "KITTI", "SYNTHIA": "Cityscapes"}
CLASSES = {
    "VKITTI": range(14),
    "KITTI": list(KITTI2VKITTI.values()),
    "SYNTHIA": range(23),
    "Cityscapes": list(CITYSCAPES2SYNTHIA.values()),
}
IDX2LABEL = {"VKITTI": VirtualKITTI.idx2label, "SYNTHIA": SYNTHIA.idx2label}


torch.backends.cudnn.benchmark = True


def eval_model(model, loader, labels, device, stats, remap_target_fn=None):
    """ Evals a model on a given dataset
    """

    for batch_cnt, (images, targets) in enumerate(loader):

        # cuda
        images = images.to(device)
        targets = targets.to(device)
        if remap_target_fn is not None:
            targets = remap_target_fn(targets)

        # inference
        outputs = model(images)["out"]
        predictions = outputs.argmax(1)

        # compute ious per example
        # TODO: both computing the confusion matrix and the IoU are extremely
        # inefficient
        stats(predictions, targets, labels=labels)

        # compute loss (don't know how just yet)
        # TODO: fix the class discrepancy between datasets
        # losses.append(F.cross_entropy(outputs, targets)
        # losses.append(0.42)

        if batch_cnt % 10 == 0:
            print("*", end="")

    print("\n", stats)

    return stats.log()


def get_data(dset_name, batch_size, workers):
    """ Configure a loader. """
    opt = YamlNamespace(
        model=YamlNamespace(model="deeplabv3_resnet101"),  # for transformation
        dataset=YamlNamespace(
            name=dset_name,
            root=NAME2PATH[dset_name],
            styled=None,
            augmentation=YamlNamespace(strong=False),
        ),
    )
    _, val_dset = get_datasets(opt, only_val=True)
    print("VALID: ", val_dset)
    print("\nJOINT TRANSFORM: ", val_dset._joint_transforms)

    loader = DataLoader(
        val_dset,
        batch_size=batch_size,
        num_workers=int(workers),
        pin_memory=True,
    )
    print("WORKERS: ", loader.num_workers)
    print("\nLOADER iterations: ", len(loader))
    return loader


def main(cmdl):
    """ Configure objects, load checkpoints, call eval_model
    """
    print("Configuration:\n", config_to_string(cmdl))
    device = torch.device(cmdl.device)
    cfg = read_config(f"{cmdl.experiment_path}/cfg.yml")
    print("Experiment:\n", config_to_string(cfg))

    # load checkpoint paths
    experiment_path = Path(cmdl.experiment_path)
    if experiment_path.is_file():
        checkpoints = [experiment_path]
    else:
        checkpoints = list(experiment_path.glob("*.pth"))
    checkpoints = [(p, int(p.stem.split("_")[1][1:])) for p in checkpoints]
    checkpoints = sorted(checkpoints, key=lambda t: t[1])
    print(f"Found {len(checkpoints)} checkpoints.")

    # filter checkpoints
    if cmdl.checkpoint:
        print(f"Filtering after step={cmdl.checkpoint} checkpoint.\n")
        checkpoints = [c for c in checkpoints if c[1] == cmdl.checkpoint]
        assert checkpoints, f"Checkpoint not found: {checkpoints}."

    # get data loader
    dset_name = cfg.dataset.name if cmdl.same else SYNTH2REAL[cfg.dataset.name]
    val_loader = get_data(dset_name, cmdl.batch_size, cmdl.workers)
    print(f"Evaluating on {len(val_loader)} batches.")

    # get model
    model = torch.nn.DataParallel(get_model(cfg))

    # configure the remapping
    if cmdl.same:
        remap_target_fn = None
    else:
        mapping_dict = (
            KITTI2VKITTI if cfg.dataset.name == "VKITTI" else CITYSCAPES2SYNTHIA
        )
        remap_target_fn = partial(remap_target, mapping=mapping_dict)
        print(f"\nTargets will be remapped as follows:\n{mapping_dict}.")

    # configure the labels to evaluate on.
    labels = CLASSES[dset_name]
    print(f"\nModel will be evaluated on the following labels\n{labels}.")

    evals = []
    checkpoints = checkpoints[:2]
    for i, (checkpoint_path, step) in enumerate(checkpoints):
        print(f"\n[{i:2d}] Loading {checkpoint_path}.\n")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        stats = StatsCollector(
            [
                IoUMetric(idx2label=IDX2LABEL[cfg.dataset.name]),
                ConfusionMatrix(IDX2LABEL[cfg.dataset.name]),
            ]
        )

        with torch.no_grad():
            stats = eval_model(
                model,
                val_loader,
                labels,  # !! pick the labels to evaluate on
                device,
                stats,
                remap_target_fn=remap_target_fn,
            )

            # save eval
            evals.append(
                {
                    "dset": dset_name,
                    "step": step,
                    "checkpoint_id": i,
                    "checkpoint": checkpoint_path,
                    "stats": stats,
                    "idx2label": IDX2LABEL[cfg.dataset.name],
                    "experiment_path": cmdl.experiment_path,
                }
            )
        torch.save(
            evals, f"{cmdl.experiment_path}/offline_eval_{dset_name}.pkl"
        )

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="ShapeBias offline eval.")
    PARSER.add_argument(
        "experiment_path",
        type=str,
        help="Path of the experiment containing saved models.",
    )
    PARSER.add_argument(
        "-c",
        "--checkpoint",
        default=0,
        type=int,
        help="Specify the checkpoint to load in terms of step index. eg.: 5000.",
    )
    PARSER.add_argument(
        "--same",
        action="store_true",
        help=(
            "Evaluates a model on the same validation dataset as the one it "
            "was trained on. Otherwise it evaluates on the real counterpart."
        ),
    )
    PARSER.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device to work do inference on. Default is `cuda`.",
    )
    PARSER.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Batch size. Default is 4.",
    )
    PARSER.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="PyTorch Dataloader number of processes. Default is 4.",
    )
    main(PARSER.parse_args())
