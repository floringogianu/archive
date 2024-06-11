""" Mostly factories for datasets, optimizers and schedulers.
"""
import rlog
import torch
from albumentations.pytorch import ToTensor

import src.datasets as D
from src.deeplab_v2 import DeepLabV2
from src.io_utils import YamlNamespace
from src.msc import MSC
from src.utils import PolynomialLR, get_optimizer, set_custom_lr

NAME2DATASET = {
    "VKITTI": D.VirtualKITTI,
    "KITTI": D.KITTI,
    "Cityscapes": D.CityscapesWrapper,
    "SYNTHIA": D.SYNTHIA,
}
NAME2PATH = {
    "VKITTI": "./data/vkitti/",
    "KITTI": "./data/rkitti/",
    "SYNTHIA": "./data/SYNTHIA_RAND_CITYSCAPES/",
    "Cityscapes": "./data/Cityscapes-HalfSize",
}
SYNTH2REAL = {"VKITTI": "KITTI", "SYNTHIA": "Cityscapes"}


def get_datasets(opt, only_val=False, with_target_dset=False):
    """ Configures and returns train and validation datasets.
    """

    try:
        dataset_class = NAME2DATASET[opt.dataset.name]
    except KeyError:
        raise NotImplementedError(f"{opt.dataset.name} not implemented.")

    val_set = dataset_class(
        opt.dataset.root,
        split="val" if opt.dataset.name != "KITTI" else "train",
        styled=None,
        joint_transforms=D.get_joint_transform(opt, "val"),
        transform=ToTensor(),
        target_transform=None,
        target_type="semantic",
    )

    if only_val:
        return val_set

    train_set = dataset_class(
        opt.dataset.root,
        split="train",
        styled=opt.dataset.styled,
        style_ratio=opt.dataset.style_ratio,
        joint_transforms=D.get_joint_transform(opt, "train"),
        transform=ToTensor(),
        target_transform=None,
        target_type="semantic",
    )

    if with_target_dset:
        target_dset_name = SYNTH2REAL[opt.dataset.name]
        dataset_class = NAME2DATASET[target_dset_name]
        target_set = dataset_class(
            NAME2PATH[target_dset_name],
            split="val" if target_dset_name != "KITTI" else "train",
            styled=None,
            joint_transforms=D.get_joint_transform(
                YamlNamespace(  # for transformation
                    model=YamlNamespace(model=opt.model.model),
                    dataset=YamlNamespace(
                        name=target_dset_name,
                        augmentation=YamlNamespace(strong=False),
                    ),
                ),
                "val",
            ),
            transform=ToTensor(),
            target_transform=None,
            target_type="semantic",
        )
        return train_set, val_set, target_set
    return train_set, val_set


def reset_modules(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass


def get_model(opt):
    """ Configures and returns a model.
    """
    model_name = opt.model.model
    # Configure model
    if model_name == "deeplabv2_resnet101":
        kwargs = opt.model.__dict__
        kwargs["n_classes"] = opt.dataset.num_classes
        model = torch.hub.load(**kwargs)
    elif model_name == "deeplabv2_vgg16":
        model = MSC(
            DeepLabV2(
                opt.dataset.label_no,
                pretrained=opt.model.pretrained,
                pretrained_backbone=opt.model.pretrained_backbone,
            ),
            scales=[0.5, 0.75],
        )
    elif model_name == "deeplabv3_resnet101":
        # opt.dataset.num_classes should specify:
        # - 14 (VKitti)
        # - 23 (SYNTHIA)
        opt.model.num_classes = opt.dataset.num_classes
        model = torch.hub.load(**opt.model.__dict__)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ResNet's default initialization produces some very large activations.
    # Therefore we use PyTorch default initialization.
    if not opt.model.pretrained_backbone:
        rlog.getLogger(opt.experiment).warning("Reseting backbone!")
        model.backbone.apply(reset_modules)

    return model
