import numpy as np
import torch
import torch.nn as nn


class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class Maxout(Activation):
    def __init__(self, num_units, axis=-1):
        super(Maxout, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        return maxout(x, self.num_units, self.axis)

    def extra_repr(self):
        return "num_units: {}".format(self.num_units)


class MaxMin(Activation):
    def __init__(self, num_units, axis=-1):
        super(MaxMin, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        maxes = maxout(x, self.num_units, self.axis)
        mins = minout(x, self.num_units, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return "num_units: {}".format(self.num_units)


def process_maxmin_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError(
            "number of features({}) is not a "
            "multiple of num_units({})".format(num_channels, num_units)
        )
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis + 1, num_channels // num_units)
    return size


def maxout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]


class GroupSort(nn.Module):
    def __init__(self, group_size, axis=-1, new_impl=False):
        super(GroupSort, self).__init__()
        self.group_size = group_size
        self.axis = axis
        self.new_impl = new_impl

    def lipschitz_constant(self):
        return 1

    def forward(self, x):
        group_sorted = group_sort(x, self.group_size, self.axis, self.new_impl)
        return group_sorted

    def extra_repr(self):
        return "group_size={group_size}, axis={axis}".format(**self.__dict__)


def group_sort(x, group_size, axis=-1, new_impl=False):
    if new_impl and group_size == 2:
        a, b = x.split(x.size(axis) // 2, axis)
        a, b = torch.max(a, b), torch.min(a, b)
        return torch.cat([a, b], dim=axis)
    shape = list(x.shape)
    num_channels = shape[axis]
    assert num_channels % group_size == 0
    shape[axis] = num_channels // group_size
    shape.insert(axis, group_size)
    if axis < 0:
        axis -= 1
    assert shape[axis] == group_size
    return x.view(*shape).sort(dim=axis)[0].view(*x.shape)
