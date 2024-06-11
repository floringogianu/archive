""" Some util functions and classes.
"""
import gc
import math
from collections import defaultdict
from typing import Iterable, Optional, Tuple

import rlog
import psutil
import torch
import torch.optim as optim
import torchvision.transforms as T
from IPython.lib.pretty import pretty
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import _LRScheduler

# this monkeypatches torch.optim
import src.radam  # pylint: disable=unused-import


INITIAL_MEM = None


def gc_stats(topk=None):
    """ Call it when playgrounds says OOMKilled.
    """
    obj_cnt = defaultdict(int)
    max_key_len = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or torch.is_storage(obj):
            try:
                shape = tuple(obj.size())
            except TypeError:
                shape = obj.size()

            key = f"[T]: {pretty(type(obj))}, {shape}"
            obj_cnt[key] += 1
        else:
            key = pretty(type(obj))
            obj_cnt[key] += 1
        max_key_len = max(max_key_len, len(key))

    sorted_cnt = sorted(obj_cnt.items(), key=lambda kv: kv[1], reverse=True)
    th_objects = {k: v for k, v in sorted_cnt if "[T]" in k}
    py_objects = {k: v for k, v in sorted_cnt if "[T]" not in k}

    header = "\n{:{width}} |   {:6}".format("Torch", "Count", width=max_key_len)
    sep = "-" * len(header)
    table = f"{sep}\n{header}\n{sep}\n"

    # print torch objects
    for i, (k, v) in enumerate(th_objects.items()):
        table += "{:{width}} |   {:6d}\n".format(k, v, width=max_key_len)
        if topk is not None and i == topk:
            table += f"... {len(th_objects) - i} tensors not displayed ...\n"
            break

    # print the other python objects
    header = "{:{width}} |    {:6}".format("Other", "Count", width=max_key_len)
    table += f"\n{sep}\n{header}\n{sep}\n"
    for i, (k, v) in enumerate(py_objects.items()):
        table += "{:{width}} |   {:6d}\n".format(k, v, width=max_key_len)
        if topk is not None and i == topk:
            table += f"... {len(py_objects) - i} py objects not displayed ...\n"
            break
    table += sep + "\n"
    table += "{:{width}} |   {:6d}\n".format(
        "Tensors Allocated", sum(th_objects.values()), width=max_key_len
    )
    table += "{:{width}} |   {:6d}\n".format(
        "Total Allocated", sum(obj_cnt.values()), width=max_key_len
    )
    table += sep
    return table


def wtf_mem(topk=None):
    global INITIAL_MEM

    table = gc_stats(topk=topk)
    sep = "-" * len(table.split("\n")[0])

    sep = "-" * 30
    vram = psutil.virtual_memory()
    vram_dict = {
        "used": vram.used,
        "active": vram.active,
        "inactive": vram.inactive,
        "free": vram.free,
        "buffers": vram.buffers,
        "shared": vram.shared,
        "cached": vram.cached,
    }

    if INITIAL_MEM is None:
        INITIAL_MEM = vram_dict

    table += "\nVirtual Memory Stats:\n"
    for k, v in vram_dict.items():
        init_val = INITIAL_MEM[k]
        table += "{:10}: {:10.3f} GiB  |  {:10.3f} GiB  |  {:10.3f} MiB\n".format(
            k,
            v / (1024 ** 3),
            init_val / (1024 ** 3),
            (v - init_val) / (1024 ** 2),
        )
    table += sep + "\n"
    return table


def is_iterable(obj):
    """ Checks if an object is iterable """
    return any(hasattr(obj, atr) for atr in ("__iter__", "__getitem__"))


#  Metrics


def get_ious(outputs, targets, labels: Optional[Iterable] = None) -> tuple:
    """ Compute mean IoU.

    Args:
        outputs (torch.ByteTensor): N*H*W with predicted label indices.
        targets (torch.ByteTensor): N*H*W with target label indices.
        labels (Optional(Iterable)): a list of labels for which we compute
            IoU. If we compute the IoU for the labels available in the
            targets.

    Returns:
        float: Mean IoU over the entire batch.
        dict: Mean IoU per class over the entire batch.
    """
    assert outputs.shape == targets.shape, "X and Y should be of same shape."
    assert outputs.shape[0] == 1, "Inputs require a batch dimension of 1."
    ious = {}
    out = outputs.view(-1)
    tgt = targets.view(-1)
    if labels is not None:
        assert is_iterable(labels), "`labels` must be an iterable."
    else:
        labels = set(torch.cat([out.unique(), tgt.unique()]).cpu().numpy())

    for cls_idx in labels:
        out_idxs = out == cls_idx
        tgt_idxs = tgt == cls_idx
        intersection = (tgt_idxs & out_idxs).sum()
        union = (out_idxs | tgt_idxs).sum()
        if union == 0:
            ious[cls_idx] = float("nan")  # no groundtruth for this class
        else:
            ious[cls_idx] = (intersection.float() / union).item()
    ious_ = torch.tensor(list(ious.values()))  # pylint: disable=not-callable
    miou = ious_[~torch.isnan(ious_)].mean().item()
    return miou, ious


class IoUMetric:
    """ Computes mean IoU and per label IoU and displays them nicely.
    """

    def __init__(self, idx2label=None):
        self._idx2label = idx2label
        self._mean_ious = []  # averages over idx2label
        self._iou_per_label = defaultdict(list)

    def step(self, outputs, targets, labels, **kwargs):
        """ This expects a baches """
        for output, target in zip(outputs, targets):
            miou, ious = get_ious(
                output.unsqueeze(0), target.unsqueeze(0), labels
            )

            self._mean_ious.append(miou)
            for label, iou in ious.items():
                self._iou_per_label[label].append(iou)

    def log(self) -> Tuple[float, dict]:
        return (
            torch.tensor(self._mean_ious).mean().item(),
            {
                label: torch.tensor(ious).mean().item()
                for label, ious in self._iou_per_label.items()
            },
        )

    def __str__(self):
        miou = torch.tensor(self._mean_ious).mean().item()
        iou_per_idx = {
            label: torch.tensor([x for x in ious if str(x) != "nan"])
            .mean()
            .item()
            for label, ious in self._iou_per_label.items()
        }

        if self._idx2label:
            header = f"  ID  | Label          |    IoU    "
        else:
            header = f"  ID  |   IoU   "
        line = "-" * len(header) + "\n"
        header = f"{line}{header}\n{line}\n"
        body = ""
        for idx, iou in sorted(iou_per_idx.items(), key=lambda kv: kv[0]):
            if self._idx2label is not None:
                label = self._idx2label[idx]
                body += f"  {idx:2d}  | {label:<12}   |   {iou*100:5.1f}  \n"
            else:
                body += f"  {idx:2d}  |   {iou*100:5.1f}  \n"
        body += line
        body += f" mean IoU: {miou*100:2.1f}\n"
        body += line
        # print(torch.tensor(list(iou_per_idx.values())).mean().item())
        return f"\n{header}{body}"


class ConfusionMatrix:
    """ Incrementally computes a confusion matrix.
    """

    def __init__(self, idx2label):
        self._idx2label = idx2label
        self._label_num = N = len(idx2label)
        self._cm = torch.zeros((N, N), dtype=torch.int64)

    def step(self, output, target, **kwargs):
        out, tgt = output.view(-1).cpu().numpy(), target.view(-1).cpu().numpy()
        cm = confusion_matrix(tgt, out, labels=range(self._label_num))
        self._cm += torch.from_numpy(cm)

    def log(self):
        return self._cm.numpy()

    def __str__(self):
        return str(self._cm)


class StatsCollector:
    def __init__(self, metrics):
        self._metrics = metrics

    def step(self, output, target, **kwargs):
        for metric in self._metrics:
            metric.step(output, target, **kwargs)

    def log(self):
        return {type(m).__name__: m.log() for m in self._metrics}

    def __call__(self, *args, **kwargs):
        self.step(*args, **kwargs)

    def __str__(self):
        consolidated = ""
        for metric in self._metrics:
            metric_name = type(metric).__name__
            consolidated += f"{metric_name}:\n\n"
            consolidated += f"{metric}\n"
        return consolidated


# Optimization utils


def get_optimizer(model, opt):
    """ Configures and returns an optimizer and a scheduler.
    """
    # Optimizer
    optim_name = getattr(optim, opt.optim.name)
    optimizer = optim_name(params=model.parameters(), **opt.optim.args.__dict__)

    # Scheduler
    if hasattr(opt, "scheduler"):
        if opt.scheduler.name == "PolynomialLR":
            scheduler = PolynomialLR(
                optimizer=optimizer, **opt.scheduler.args.__dict__
            )
        elif opt.scheduler.name == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, **opt.scheduler.args.__dict__
            )
    else:
        scheduler = None
    return optimizer, scheduler


class PolynomialLR(_LRScheduler):
    """ From github.com/kazuto1011/deeplab-pytorch/
    """

    def __init__(self, optimizer, step_size, iter_max, power, last_epoch=-1):
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]

    def __str__(self):
        return f"PolyLR iter_max={self.iter_max}, step_size={self.step_size}"


def get_params(model, key):
    """ Adapted from github.com/kazuto1011/deeplab-pytorch/
    """
    # For Dilated FCN
    if key == "1x":
        for module_name, module in model.named_modules():
            if any([x in module_name for x in ("backbone", "layer")]):
                if isinstance(module, torch.nn.Conv2d):
                    for p in module.parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for module_name, module in model.named_modules():
            if "aspp" in module_name:
                if isinstance(module, torch.nn.Conv2d):
                    yield module.weight
    # For conv bias in the ASPP module
    if key == "20x":
        for module_name, module in model.named_modules():
            if "aspp" in module_name:
                if isinstance(module, torch.nn.Conv2d):
                    yield module.bias


def set_custom_lr(model, optim_args):
    """ Sets multiples of the base learning rate for various layer types.
    """
    return [
        {
            "params": get_params(model, "1x"),
            "lr": optim_args.lr,
            "weight_decay": optim_args.weight_decay,
        },
        {
            "params": get_params(model, "10x"),
            "lr": 10 * optim_args.lr,
            "weight_decay": optim_args.weight_decay,
        },
        {
            "params": get_params(model, "20x"),
            "lr": 20 * optim_args.lr,
            "weight_decay": optim_args.weight_decay,
        },
    ]


def reset_optimizer_(opt, step_cnt, model, optimizer, scheduler):
    """ Reset optimizer and scheduler every `opt.step_no * opt.cyclic.breaks`
    """
    train_log = rlog.getLogger(opt.experiment + ".train")
    cycle_steps = int(math.floor(opt.step_no * opt.cyclic.breaks))
    if (step_cnt - 1) % cycle_steps == 0:
        # compute the no of steps for this cycle
        opt.scheduler.iter_max = cycle_steps
        # compute the index of the starting lr value for this cycle
        lr_idx = int((step_cnt - 1) // cycle_steps)
        # set the new learning rate
        opt.optim.args.lr = opt.cyclic.lrs[lr_idx]
        # reset the optimizer and the schedulers
        optimizer, scheduler = get_optimizer(model, opt)
        print(scheduler)
        print(opt.optim.args)
        print(opt.scheduler)
        train_log.info("New cycle. Restart optimizer @%5d steps.", step_cnt)
        train_log.info("Initial lr=%.8f.", opt.optim.args.lr)
    return optimizer, scheduler


# Visualization


def show_image(img, output, class_num=14, title="Figure"):
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(class_num)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the image
    fig = plt.figure(num=title)

    # original
    fig.add_subplot(1, 2, 1)
    plt.imshow(T.ToPILImage()(img))

    # grayscale + target
    fig.add_subplot(1, 2, 2)
    plt.imshow(T.ToPILImage()(img), cmap="gray")

    # plot the semantic segmentation
    np_shape = output.shape[1], output.shape[0]
    r = Image.fromarray(output.byte().cpu().numpy()).resize(np_shape)
    r.putpalette(colors)
    plt.imshow(r, alpha=0.8)

    plt.show()


if __name__ == "__main__":
    pass
