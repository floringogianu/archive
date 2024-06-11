""" Dataset factory.
"""
from copy import deepcopy
import math
from torch.utils.data import random_split
from torchvision import datasets as D
from torchvision import transforms as T


def get_dsets(opt, root="./data/cifar-data"):
    """ Returns configured train and test datasets.
    """
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )
    test_transform = T.Compose([T.ToTensor(), normalize])

    train_dset = D.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )

    test_dset = D.CIFAR10(root=root, train=False, transform=test_transform)

    if hasattr(opt, "warmup"):
        partition = math.floor(len(train_dset) * opt.warmup.split)
        lengths = [partition, len(train_dset) - partition]
        warmup_dset = random_split(train_dset, lengths)[0]
        return train_dset, test_dset, warmup_dset

    return train_dset, test_dset, None


def get_unaugmented(dset):
    """ Remove the augmentation from a training dataset without side effects.
    """
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    test_transform = T.Compose([T.ToTensor(), normalize])
    dset_ = deepcopy(dset)
    setattr(dset_, "transform", test_transform)
    return dset_
