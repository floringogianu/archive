""" A hodgepodge of utilities for checking stuff.

    1. dataset_stats: iterates through dataset and outputs the class idxs :)
    2. visualize_data: this one is actually handy. It opens each training
        example as seen by the model, complete with transforms. Also
        overimposes the groundtruth for easily checking that the transforms
        are not destructive.
"""
import argparse

from torch.utils.data import DataLoader

from src import get_datasets
from src.io_utils import YamlNamespace
from src.utils import show_image

NAME2PATH = {
    "VKITTI": "./data/vkitti/",
    "KITTI": "./data/rkitti/",
    "SYNTHIA": "./data/SYNTHIA_RAND_CITYSCAPES/",
    "Cityscapes": "./data/Cityscapes-HalfSize",
}


def _get_data(dset_name, styled, val, batch_size, normalized, style_ratio):
    """ Configure a loader. """
    opt = YamlNamespace(
        model=YamlNamespace(model="deeplabv3_resnet101"),  # for transformation
        dataset=YamlNamespace(
            name=dset_name,
            root=NAME2PATH[dset_name],
            styled=styled,
            style_ratio=style_ratio,
            augmentation=YamlNamespace(strong=True, normalized=normalized),
        ),
    )
    trn_dset, val_dset = get_datasets(opt)
    if val:
        print("VALID: ", val_dset)
        print("\nJOINT TRANSFORM: ", val_dset._joint_transforms)
    else:
        print("TRAIN: ", trn_dset)
        print("\nJOINT TRANSFORM: ", trn_dset._joint_transforms)

    loader = DataLoader(
        val_dset if val else trn_dset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=batch_size,
    )
    print("\nLOADER iterations: ", len(loader))
    return loader


def benchmark_inference(loader, model):


def dataset_stats(loader):
    """ Gathers some stats with respect to the target distribution. """
    all_class_idxs = set()
    for i, (_, targets) in enumerate(loader):
        all_class_idxs |= set(targets.unique().numpy())
        if i % 10 == 0:
            print(f"{i:3d}", all_class_idxs, f"len={len(all_class_idxs)}")

    print("\n", all_class_idxs)


def visualize_data(loader):
    """ Loop through the loader and visualize the data. """
    for images, targets in loader:
        show_image(
            images[0],
            targets[0],
            class_num=loader.dataset.label_no,
            title=type(loader.dataset).__name__,
        )


def main(cmdl):
    loader = _get_data(
        dset_name=cmdl.dataset,
        styled=cmdl.styled,
        batch_size=cmdl.batch_size,
        val=cmdl.val,
        style_ratio=cmdl.style_ratio,
        normalized=True,
    )

    eval(cmdl.fn_name)(loader)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="ShapeBias misc stuff.")
    PARSER.add_argument(
        "-f",
        "--fn_name",
        type=str,
        default="visualize_data",
        help="Name of the function to execute. Defaults to `visualize_data`",
    )
    PARSER.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="VKITTI",
        help="Dataset to load: VKITTI, SYNTHIA, KITTI, Cityscapes.",
    )
    PARSER.add_argument(
        "-s",
        "--styled",
        type=str,
        default=None,
        help="Style strategy: None, `all`, `picked`. Defaults to None.",
    )
    PARSER.add_argument(
        "-r",
        "--style-ratio",
        type=str,
        default=1.0,
        help="Style ratio: 0.5, 1.0. Used only if `styled` is different than "
        + "None. Defaults to 1.0.",
    )
    PARSER.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Default is 1.",
    )
    PARSER.add_argument(
        "--val",
        action="store_true",
        help="Load the validation split instead of the train one.",
    )
    main(PARSER.parse_args())
