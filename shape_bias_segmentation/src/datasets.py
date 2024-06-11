""" Dataset implementations.
"""


__all__ = [
    "KITTI",
    "VirtualKITTI",
    "CityscapesWrapper",
    "SYNTHIA",
    "KITTI2VKITTI",
    "CITYSCAPES2SYNTHIA",
    "remap_target",
]


import csv
from collections import namedtuple
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import albumentations as T
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.cityscapes import Cityscapes


# A mapping of the ten classes that intersect between KITTI and VirtualKITTI
# according to "Learning Semantic Segmentations from Synthetic Data..."
# https://arxiv.org/abs/1812.05040
KITTI2VKITTI = {
    7: 5,  # road
    11: 0,  # building
    17: 4,  # pole
    19: 8,  # traffic light
    20: 9,  # traffic sign
    21: 13,  # vegetation
    22: 7,  # terrain
    23: 6,  # sky
    26: 1,  # car
    27: 11,  # truck
}
# Mapping of the 16 classes that intersect between SYNTHIA and Cityscapes.
# I am using the Cityscapes ids (not train_ids) from here:
# github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# And the SYNTHIA labels from Readme file in the SINTHIA-RAND-CITYSCAPES data
# http://synthia-dataset.net/downloads/
CITYSCAPES2SYNTHIA = {
    # 0: 0,  #  misc
    23: 1,  # sky
    11: 2,  # building
    7: 3,  # road
    8: 4,  # sidewalk
    13: 5,  # fence
    21: 6,  # vegetation
    17: 7,  # pole
    26: 8,  # car
    20: 9,  # traffic sign
    24: 10,  # pedestrian / person
    33: 11,  # bicycle
    32: 12,  # motorbike
    # 9: 13,  # parkingslot
    # None: 14,  # Road-work ??
    19: 15,  # traffic light
    # 16: 22,  # terrain
    25: 17,  # rider
    # 27: 18,  # truck
    28: 19,  # bus
    # 31: 20,  # train
    12: 21,  # wall
    # None: 22,  # lane-marking
}


def remap_target(target, mapping):
    """ Maps label ids to other label ids in a target tensor.
    """
    target_ = target.clone()
    for k, v in mapping.items():
        target[target_ == k] = v
    return target


def cv_loader(path):
    """ OpenCV image loader.
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


class KITTI(VisionDataset):
    """ KITTI dataset for semantic segmentation
    """

    # KITTI segmentation follows the conventions in Cityscapes
    # Based on https://github.com/mcordts/cityscapesScripts
    KITTIClass = namedtuple(
        "KITTIClass",
        [
            "name",
            "id",
            "train_id",
            "category",
            "category_id",
            "has_instances",
            "ignore_in_eval",
            "color",
        ],
    )

    # fmt: off
    classes = [
        KITTIClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        KITTIClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        KITTIClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        KITTIClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        KITTIClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        KITTIClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        KITTIClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        KITTIClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        KITTIClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        KITTIClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        KITTIClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        KITTIClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        KITTIClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        KITTIClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        KITTIClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        KITTIClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        KITTIClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        KITTIClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        KITTIClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        KITTIClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        KITTIClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        KITTIClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        KITTIClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        KITTIClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        KITTIClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        KITTIClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        KITTIClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        KITTIClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        KITTIClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        KITTIClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        KITTIClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        KITTIClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        KITTIClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        KITTIClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        KITTIClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    # fmt: on
    def __init__(  # pylint: disable=bad-continuation,too-many-arguments
        self,
        root,
        split="train",
        joint_transforms=None,
        transform=None,
        target_transform=None,
        loader=cv_loader,
        target_type="semantic",
        **kwargs,
    ):
        """ KITTI dataset constructor.

        Args:
            root (str): Path to the `data_semantics` folder.
            split (str, optional): The dataset split to use, `train` or `test`.
                Only `train` has groundtruths. Defaults to `train`.
            joint_transforms (callable, optional): A transformation that
                applies to both images and targets. Defaults to None.
            transform (callable, optional): A transformation that applies
                only to images. Defaults to None.
            target_transform (callable, optional): A transformation that
                applies only to targets. Defaults to None.
            loader (callable, optional): Function that loads an image.
                Defaults to `pil_loader`.
            target_type (str, optional): Type of target to use, `instance`,
                `semantic` o r`semantic_rgb`. Defaults to 'semantic'.
        """
        super(KITTI, self).__init__(root)
        self._split = split = split + "ing"
        self._joint_transforms = joint_transforms
        self._img_transform = transform
        self._tgt_transform = target_transform
        self._loader = loader
        self._target_type = target_type

        img_path = Path(root) / split / "image_2"
        tgt_path = Path(root) / split / target_type
        self._img_paths = [f for f in img_path.glob("**/*.png")]
        self._tgt_paths = [f for f in tgt_path.glob("**/*.png")]

    def __getitem__(self, idx):

        # on test split we only load images
        if self._split == "testing":
            image = self._loader(self._img_paths[idx].as_posix())
            if self._img_transform:
                image = self._img_transform(image)
            return image

        # Load image and target, should always return a PIL Image irrespective
        # of the actual loader (opencv, PIL, etc.)
        image = self._loader(self._img_paths[idx].as_posix())
        target = self._loader(self._tgt_paths[idx].as_posix())
        if self._target_type != "semantic_rgb":
            target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

        # Apply stochastic transforms on both image and groundtruth,
        # for example RandomCrop
        if self._joint_transforms:
            aug = self._joint_transforms(image=image, mask=target)
            image, target = aug["image"], aug["mask"]

        # Apply other individual transforms on images
        if self._img_transform:
            image = self._img_transform(image=image)["image"]

        # Apply other individual transforms on groundtruths
        if self._tgt_transform:
            target = self._tgt_transform(target)

        return image, target

    def __len__(self):
        return len(self._img_paths)


class RGB2idx:
    """ Fast mapping from H*W*C semantic segmentation groundtruths
    based on pixel values to H*W filled with label indices.

    We map each RGB to a scalar value in 255^3 space and each of these
    scalars are mapped to the label index (eg. Building: 0, Car: 1).

    This allows for vectorized mapping for large images.
    """

    def __init__(self, rgb2idx):
        self.color_map = torch.zeros(256 ** 3, dtype=torch.long) - 1
        self.mapper = torch.tensor([256 ** 2, 256, 1], dtype=torch.long)
        for rgb, idx in rgb2idx.items():
            rgb_ = torch.tensor(rgb, dtype=torch.long).dot(self.mapper)
            self.color_map[rgb_.item()] = idx

    def __call__(self, np_img):
        th_img = torch.from_numpy(np_img).long()
        th_img_ = th_img.matmul(self.mapper)
        return self.color_map[th_img_].long()


def get_rgb2label(path):
    """ Reads all the csv files in vkitti_scenegt folder and consolidates
    the label - rgb encodings for all the labels.
    """
    csv_paths = [p for p in path.glob("*.txt") if "README" not in p.name]
    codes = []
    for csv_path in csv_paths:
        with open(str(csv_path)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            next(csv_reader)  # ignore head
            for row in csv_reader:
                codes.append(",".join(row))

    # the RGB encoding for each label or instance remains consistent
    # across worlds and variations.
    codes = sorted(set(codes))

    rgb2label = {}
    for code in codes:
        label, r, g, b = code.split(",")  # pylint: disable=invalid-name
        rgb = tuple(map(int, [r, g, b]))
        # we consolidate the instances of Car and Van into their respective
        # labels. (Van:01 -> Van)
        rgb2label[rgb] = label.split(":")[0]
    return rgb2label


class VirtualKITTI(VisionDataset):
    """ Virtual KITTI dataset for semantic segmentation.

        This class assumes v1.3.1 version of the dataset. See
        https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds/
        for a description of the dataset structure.

        Computed statistics:
        mean = [0.3878, 0.4225, 0.3700]
        std = [0.2130, 0.2212, 0.2472]
    """

    mean = [0.3878, 0.4225, 0.3700]
    std = [0.2130, 0.2212, 0.2472]
    worlds = ["0001", "0002", "0006", "0018", "0020"]
    variations = [
        "clone",
        "15-deg-right",
        "15-deg-left",
        "30-deg-left",
        "30-deg-right",
        "morning",
        "sunset",
        "overcast",
        "fog",
        "rain",
    ]
    label2idx = {
        "Building": 0,
        "Car": 1,
        "GuardRail": 2,
        "Misc": 3,
        "Pole": 4,
        "Road": 5,
        "Sky": 6,
        "Terrain": 7,
        "TrafficLight": 8,
        "TrafficSign": 9,
        "Tree": 10,
        "Truck": 11,
        "Van": 12,
        "Vegetation": 13,
    }
    idx2label = {v: k for k, v in label2idx.items()}

    def __init__(  # pylint: disable=bad-continuation,too-many-arguments
        self,
        root,
        split="train",
        joint_transforms=None,
        transform=None,
        target_transform=None,
        loader=cv_loader,
        targets_as_indices=True,
        styled=None,
        style_ratio=1.0,
        **kwargs,
    ):
        """ VKITTI constructor.

        Args:
            too ([type]): [description]
            root (str): Path to the root containing the data.
            split (str, optional): Wether loading train or validation data. Defaults to "train".
            styled ((str, None), optional): Type of styled data: None | all |
                picked |. None for only loading the original data. `all` for
                loading all the styles. `picked` for loading a number of
                handpicked styles. Defaults to None.
            style_ratio (float): Ratio of styled images in the dataset.
                Accepted values: {0.5, 1.0}. Is ignored if `styled=None`.
                Defaults to 1.0.
        """
        super(VirtualKITTI, self).__init__(root)
        self._split = split
        self._img_transform = transform
        self._tgt_transform = target_transform
        self._loader = loader
        self._joint_transforms = joint_transforms
        self._styled = styled
        self._style_ratio = style_ratio if styled else 0.0
        self._rgb2idx_transform = None

        assert styled in (
            None,
            "all",
            "picked",
        ), f"`styled` can be either None, `all` or `picked` not {styled}."
        assert style_ratio in (
            0.5,
            1.0,
        ), f"`style_ratio` can be either 0.5 or 1.0 for now not {style_ratio}"

        styled_opts = {
            "all": "stylized_vkitti_1.3.1_rgb",
            "picked": "handpicked_stylized_vkitti_1.3.1_rgb",
        }

        root = Path(root)
        img_path = root / "vkitti_1.3.1_rgb"
        tgt_path = root / "vkitti_1.3.1_scenegt"
        sty_path = root / styled_opts[styled] if styled else None
        print("Styled path: ", sty_path)

        self._img_paths, self._tgt_paths = get_vkitti_split(
            img_path,
            tgt_path,
            split,
            sty_path=sty_path,
            style_ratio=self._style_ratio,
        )

        self._rgb2idx = {
            rgb: VirtualKITTI.label2idx[label]
            for rgb, label in get_rgb2label(tgt_path).items()
        }

        if targets_as_indices:
            self._rgb2idx_transform = RGB2idx(self._rgb2idx)

        assert len(self._img_paths) == len(
            self._tgt_paths
        ), "Number of images and their targets don't match!"

    def __getitem__(self, idx):
        # Load image and target.
        # Can return a PIL or an OpenCV image, depending on the loader.
        image = self._loader(self._img_paths[idx].as_posix())
        target = self._loader(self._tgt_paths[idx].as_posix())

        # Apply stochastic transforms on both image and groundtruth,
        # for example RandomCrop
        if self._joint_transforms:
            aug = self._joint_transforms(image=image, mask=target)
            image, target = aug["image"], aug["mask"]

        # Apply other individual transforms on images
        if self._img_transform:
            image = self._img_transform(image=image)["image"]

        # Apply other individual transforms on groundtruths
        if self._tgt_transform:
            target = self._tgt_transform(target)

        # RGB to label idx
        if self._rgb2idx_transform:
            target = self._rgb2idx_transform(target)

        return image, target

    @property
    def rgb2idx(self):
        """ Return a mapping from RGB values to label indices. """
        return self._rgb2idx

    @property
    def label_no(self):
        return len(VirtualKITTI.label2idx)

    def __len__(self):
        return len(self._img_paths)

    def __str__(self):
        return "{dset}\n\tStyled: {styled}\n\tRatio: {ratio}".format(
            dset=super().__str__(), styled=self._styled, ratio=self._style_ratio
        )


class CityscapesWrapper(Cityscapes):
    """ Wrapper for the PyTorch implementation of Cityscapes.
    """

    label2idx = {c.name: c.id for c in Cityscapes.classes}
    idx2label = {v: k for k, v in label2idx.items()}

    def __init__(  # pylint: disable=bad-continuation,too-many-arguments
        self,
        root,
        target_type="instance",
        split="train",
        joint_transforms=None,
        transform=None,
        target_transform=None,
        **kwargs,
    ):
        super(CityscapesWrapper, self).__init__(
            root, target_type=target_type, split=split
        )
        self._img_transform = transform
        self._tgt_transform = target_transform
        self._joint_transforms = joint_transforms

    def __getitem__(self, idx):
        # Load image and target.
        # Can return a PIL or an OpenCV image, depending on the loader.
        image, target = super(CityscapesWrapper, self).__getitem__(idx)
        image = np.array(image)
        target = np.array(target)

        # Apply stochastic transforms on both image and groundtruth,
        # for example RandomCrop
        if self._joint_transforms:
            aug = self._joint_transforms(image=image, mask=target)
            image, target = aug["image"], aug["mask"]

        # Apply other individual transforms on images
        if self._img_transform:
            image = self._img_transform(image=image)["image"]

        # Apply other individual transforms on groundtruths
        if self._tgt_transform:
            target = self._tgt_transform(target)
        target = torch.tensor(target, dtype=torch.long)

        return image, target

    @property
    def label_no(self):
        return len(Cityscapes.classes)


class SYNTHIA(VisionDataset):
    """ SYNTHIA dataset for semantic segmentation.

        Channel statistics not yet computed.
    """

    label2idx = {
        "void": 0,
        "sky": 1,
        "Building": 2,
        "Road": 3,
        "Sidewalk": 4,
        "Fence": 5,
        "Vegetation": 6,
        "Pole": 7,
        "Car": 8,
        "Traffic sign": 9,
        "Pedestrian": 10,
        "Bicycle": 11,
        "Motorcycle": 12,
        "Parking-slot": 13,
        "Road-work": 14,
        "Traffic light": 15,
        "Terrain": 16,
        "Rider": 17,
        "Truck": 18,
        "Bus": 19,
        "Train": 20,
        "Wall": 21,
        "Lanemarking": 22,
    }
    idx2label = {v: k for k, v in label2idx.items()}

    def __init__(  # pylint: disable=bad-continuation,too-many-arguments
        self,
        root,
        split="train",
        styled=None,
        joint_transforms=None,
        transform=None,
        target_transform=None,
        loader=cv_loader,
        **kwargs,
    ):
        """ SYNTHIA constructor.

        Args:
            too ([type]): [description]
            root (str): Path to the root containing the data.
            split (str, optional): Wether loading train or validation data. Defaults to "train".
            styled ((str, None), optional): Wether to load styled data. None
                for only loading the original data, `mixed` for both styled
                and original, `full` for only styled data. Defaults to None.
        """
        super(SYNTHIA, self).__init__(root)
        self._split = split
        self._img_transform = transform
        self._tgt_transform = target_transform
        self._loader = loader
        self._joint_transforms = joint_transforms
        self._styled = styled

        assert styled in (
            None,
            "mixed",
            "full",
        ), "`styled` can be either None, `mixed` or `full`."

        root = Path(root)
        img_path = root / "RGB"
        tgt_path = root / "GT" / "LABELS"
        sty_path = root / "RGB_stylized" if styled else None

        self._img_paths, self._tgt_paths = get_synthia_split(
            img_path, tgt_path, split, styled=styled, sty_path=sty_path
        )

        assert len(self._img_paths) == len(
            self._tgt_paths
        ), "Number of images and their targets don't match!"

    def __getitem__(self, idx):
        # Load image and target.
        # We need to use imageio because of atypical target format.
        image = np.array(pil_loader(self._img_paths[idx].as_posix()))
        target = imageio.imread(
            self._tgt_paths[idx].as_posix(), format="PNG-FI"
        )

        target = np.array(target[:, :, 0])  # first channel contains label info

        # Apply stochastic transforms on both image and groundtruth,
        # for example RandomCrop
        if self._joint_transforms:
            aug = self._joint_transforms(image=image, mask=target)
            image, target = aug["image"], aug["mask"]

        # Apply other individual transforms on images
        if self._img_transform:
            image = self._img_transform(image=image)["image"]

        # Apply other individual transforms on groundtruths
        if self._tgt_transform:
            target = self._tgt_transform(target)

        target = target.astype("int64")
        return image, target

    @property
    def label_no(self):
        return len(SYNTHIA.label2idx)

    def __len__(self):
        return len(self._img_paths)

    def __str__(self):
        return f"{super().__str__()}\n    Styled: {self._styled}"


def get_vkitti_split(
    img_path,
    tgt_path,
    split,
    test_world="0002",
    sty_path=None,
    style_ratio=None,
):
    """ Returns image and grountruth filepaths for a given split.

    The authors don't propose a split so we pick the smallest world, "0002"
    to be the validation dataset.

    Args:
        img_path (Path): Path to the folder containing val and train examples
        tgt_path (Path): Path to the folder containing val & train groundtruths
        split (bool): Train / Test switch
        test_world (str, optional): The world used for validation.
            Defaults to "0002".
        sty_path (Path, optional): Path to styled images. Defaults to None.
        style_ratio (float): Ratio of styled images in the dataset.
            Accepted values: {0.0, 0.5, 1.0}.

    Returns:
        tuple(list,list): Return a tuple of two lists, containing paths to
            images and groundtruths.
    """
    if style_ratio:
        assert sty_path, "`sty_path` can't be None if `styled` is not None."

    img_paths, tgt_paths = [], []
    if split == "train":
        worlds = [w for w in VirtualKITTI.worlds if w != test_world]
        for w in worlds:
            if style_ratio != 1.0:
                # load the originals
                img_paths += [f for f in img_path.joinpath(w).glob("**/*.png")]
                tgt_paths += [f for f in tgt_path.joinpath(w).glob("**/*.png")]
            if style_ratio != 0.0:
                # load the styles
                img_paths += [f for f in sty_path.joinpath(w).glob("**/*.png")]
                tgt_paths += [f for f in tgt_path.joinpath(w).glob("**/*.png")]
    elif split == "val":
        img_paths += [f for f in img_path.joinpath(test_world).glob("**/*.png")]
        tgt_paths += [f for f in tgt_path.joinpath(test_world).glob("**/*.png")]
    return img_paths, tgt_paths


def get_synthia_split(img_path, tgt_path, split, sty_path=None, styled=None):
    """ Returns image and groundtruth filepaths for a given split. The dataset
    authors don't propose a split so we pick last 2400 images to be test split.

    Args:
        img_path (Path): Path to the folder containing test and train examples
        tgt_path (Path): Path to the folder containing test & train groundtruths
        split (bool): Train / Test switch
        sty_path (Path, optional): Path to styled images. Defaults to None.
        styled ((str, None), optional): Wether to load styled data. None
            for only loading the original data, `mixed` for both styled
            and original, `full` for only styled data. Defaults to None.

    Returns:
        tuple(list,list): Return a tuple of two lists, containing paths to
            images and groundtruths.
    """
    if styled:
        assert sty_path, "`sty_path` can't be None if `styled` is not None."

    if split == "train":
        indices = list(range(0, 6000))
    elif split == "val":
        indices = list(range(6000, 7000))
    elif split == "test":
        indices = list(range(7000, 9400))

    img_paths, tgt_paths = [], []
    for idx in indices:
        idx_str = str(idx).zfill(7) + ".png"
        if styled != "full":
            img_paths += [img_path.joinpath(idx_str)]
            tgt_paths += [tgt_path.joinpath(idx_str)]
        if styled is not None:
            img_paths += [sty_path.joinpath(idx_str)]
            tgt_paths += [tgt_path.joinpath(idx_str)]

    return img_paths, tgt_paths


def get_joint_transform(opt, split):
    """ Function returning augmentation for several models and datasets.

        - DeepLabV2 prefers random crops of size at least 321.
        - DeepLabV3 prefers square crops of size at least 513.
        - For all the cases we employ HorizontalFlip (as in Chen et al)
            and Normalization.
        - Images are randomly rescalled within some boundaries:
        	- to satisfy model requirements
            - to provide augmentation
        - `strong` further adds augmentation.
    """
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # resize and crop depending on the model or dataset.
    if opt.model.model in ("deeplabv2_resnet101", "deeplabv2_vgg16"):
        if opt.dataset.name == "Cityscapes":
            # input should be about 1024 x 512 (from 2048x1024, we resized them)
            jtr = [T.RandomScale((0.7, 1.2), p=1)]
        elif opt.dataset.name == "SYNTHIA":
            # input should be about 1280 x 760
            jtr = [T.RandomScale((0.7, 1.2), p=1)]
        elif opt.dataset.name in ("VKITTI", "KITTI"):
            # input should be about 1242 x 375
            jtr = [T.RandomScale((0.9, 1.4), p=1)]
        else:
            raise Exception(f"Unknown dataset: {opt.dataset.name}.")

        # !! input resolution required bt DeepLab-v2
        jtr += [T.RandomCrop(321, 321)]
    elif opt.model.model == "deeplabv3_resnet101":
        if opt.dataset.name == "Cityscapes":
            # input should be about 2048 x 1024
            jtr = [T.RandomScale((0.51, 1.1), p=1)]
        elif opt.dataset.name == "SYNTHIA":
            # input should be about 1280 x 760
            jtr = [T.RandomScale((0.7, 1.3), p=1)]
        elif opt.dataset.name in ("VKITTI", "KITTI"):
            # input should be about 1242 x 375
            jtr = [T.RandomScale((1.4, 2.0), p=1)]
        else:
            raise Exception(f"Unknown dataset: {opt.dataset.name}.")

        # !! input resolution required bt DeepLab-v3
        jtr += [T.RandomCrop(513, 513)]
    else:
        raise Exception(f"Unknown model: {opt.model.model}.")

    # during training
    if split == "train":
        jtr += [T.HorizontalFlip()]  # HorizontalFlip for all cases
        if opt.dataset.augmentation.strong:
            jtr += [
                T.RGBShift(),
                T.RandomBrightnessContrast(0.5, 0.5),
                T.MotionBlur(),
                T.JpegCompression(p=0.7),
                T.ISONoise((0.03, 0.07), p=0.7),
            ]
        # the default is normalized but we might want to visualize the data
        if not hasattr(opt.dataset.augmentation, "normalized"):
            jtr += [normalize]
        elif opt.dataset.augmentation.normalized:
            jtr += [normalize]
        return T.Compose(jtr)

    # during validation
    jtr = []
    if (
        opt.dataset.name in ("VKITTI", "KITTI")
        and opt.model.model == "deeplabv3_resnet101"
    ):
        jtr = [T.Resize(513, 1699)]
    if not hasattr(opt.dataset.augmentation, "normalized"):
        jtr += [normalize]
    elif opt.dataset.augmentation.normalized:
        jtr += [normalize]
    return T.Compose(jtr)


def resize_labels(labels, size):
    """ From github.com/kazuto1011/deeplab-pytorch/
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


def calculate_dataset_moments(dataset):
    """ Calculate channel mean and std of the given dataset.
    """
    train_set_list = [
        dataset[idx][0].detach().numpy() for idx in range(len(dataset))
    ]
    stacked_train = np.stack(train_set_list)
    stacked_train = np.transpose(stacked_train, (0, 3, 2, 1))
    channel_mean = stacked_train.mean(0).mean(0).mean(0)
    stacked_vars = np.square(stacked_train - channel_mean)
    channel_std = np.sqrt(stacked_vars.mean(0).mean(0).mean(0))
    print("channel mean", channel_mean, "channel std", channel_std)
    channel_std = np.std(stacked_train, axis=(0, 1, 2))
    print("channel mean", channel_mean, "channel std", channel_std)
    print(
        np.mean(stacked_train, axis=(0, 1, 2)),
        np.std(stacked_train, axis=((0, 1, 2))),
    )

    return channel_mean, channel_std


if __name__ == "__main__":
    pass
