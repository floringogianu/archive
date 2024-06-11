import math
import torch
from torch.nn import Linear
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def main():
    max_steps = 240_000
    steps_per_epoch = 1893
    T_0 = steps_per_epoch * 2
    T_mult = 1.3

    max_lr = 0.002
    min_lr = 0.00001

    m = Linear(20, 2)
    o = SGD(m.parameters(), lr=max_lr)
    s = CosineAnnealingWarmRestarts(o, T_0=T_0, T_mult=T_mult, eta_min=min_lr)

    resets = []
    for i in range(12):
        if not resets:
            resets.append(T_0)
        else:
            resets.append(math.ceil(T_0 + resets[-1] * T_mult))
    print(resets)

    for i in range(max_steps):
        o.step()
        s.step()
        lr = "{:04d}  lr={:1.6f}".format(i, s.get_lr()[0])
        if i % int(T_0 / 2) == 0:
            lr = lr + "  <<" if i in resets else lr
            print(lr)
        elif i in resets:
            print(lr, "  <<")


if __name__ == "__main__":
    main()
