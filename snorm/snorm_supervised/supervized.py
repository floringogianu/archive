""" Supervise me while I'm snormin' around.
"""

from functools import partial
import pathlib

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src import estimators
from src.monitor import Monitor


def get_optimizer(name, parameters, **kwargs):
    """ Here we prepare the optimizer.
    """
    return getattr(optim, name)(parameters, **kwargs)


def get_scheduler(name, optimizer, **kwargs):
    """ Here we prepare the scheduler.
    """
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)


def printer(monitor):
    """ Print just a line with the accuracy and the negative log likelihood on both
        train and test.
    """
    print(
        f" Epoch {monitor.last('epoch'):3d}"
        f" | TstAcc:{monitor.last('test/acc'):6.2f}%"
        f" | TstNll:{monitor.last('test/nll'):.4f}"
        # f" | TrnAcc:{monitor.last('train/acc'):6.2f}%"
        # f" | TrnNll:{monitor.last('train/nll'):.4f}"
        # f" | SN.3:{monitor.last('snorm/3'):.4f}"
        # f" | SN.4:{monitor.last('snorm/4'):.4f}"
    )


def evaluate(model, inputs, targets, batch_size=512):
    """ Here we evaluate the model on some data.
    """
    was_training = model.training
    model.eval()

    nll, acc, ntotal = 0, 0, 0

    with torch.no_grad():
        chuncks = zip(torch.split(inputs, batch_size), torch.split(targets, batch_size))
        for xxx, ttt in chuncks:
            logits = model(xxx)
            nll += F.cross_entropy(logits, ttt, reduction="sum").item()
            acc += logits.argmax(dim=-1).eq(ttt).sum().item()
            ntotal += len(xxx)

    nll = nll / ntotal
    acc = (acc * 100) / ntotal

    if was_training:
        model.train()

    return {"acc": acc, "nll": nll}


def prepare_data(batch_size, augmentation=True, device="cpu", num_workers=2, root=None):
    """ Here we prepare the date.
    """

    if root is None:
        root = pathlib.Path(".")

    svhn_mean, svhn_std = 0.4514187438009213, 0.19929124669110956

    if augmentation:
        transforms_list = [
            transforms.RandomCrop(32, 5, padding_mode="edge"),
            transforms.ToTensor(),
            transforms.Normalize((svhn_mean,), (svhn_std,)),
            transforms.Lambda(lambda x: x[torch.randperm(3)]),
        ]
    else:
        transforms_list = [
            # transforms.RandomCrop(32, 5, padding_mode="edge"),
            transforms.ToTensor(),
            transforms.Normalize((svhn_mean,), (svhn_std,)),
            # transforms.Lambda(lambda x: x[torch.randperm(3)]),
        ]

    trn_set = datasets.SVHN(
        root.joinpath(".data"),
        split="train",
        transform=transforms.Compose(transforms_list),
        download=True,
    )
    tst_set = datasets.SVHN(
        root.joinpath(".data"), split="test", transform=None, download=True
    )

    trn_loader = DataLoader(
        trn_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
    )

    trn_data = (
        torch.from_numpy(trn_set.data)
        .float()
        .div(255.0)
        .sub(svhn_mean)
        .div(svhn_std)
        .to(device)
    )
    trn_target = torch.from_numpy(trn_set.labels).long().to(device)
    tst_data = (
        torch.from_numpy(tst_set.data)
        .float()
        .div(255.0)
        .sub(svhn_mean)
        .div(svhn_std)
        .to(device)
    )
    tst_target = torch.from_numpy(tst_set.labels).long().to(device)

    return trn_loader, (trn_data, trn_target), (tst_data, tst_target)


def run(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trn_loader, trn_set, tst_set = prepare_data(
        opts.batch_size,
        augmentation=opts.augmentation,
        device=device,
        root=opts.tmp_path if hasattr(opts, "tmp_path") else pathlib.Path("."),
    )

    model = getattr(estimators, opts.model.name)(**opts.model.args)
    # model = SVHNNet(**opts.model)
    model.to(device)
    print(model)

    optimizer = get_optimizer(parameters=model.parameters(), **opts.optimizer)
    if hasattr(opts, "scheduler"):
        scheduler = get_scheduler(optimizer=optimizer, **opts.scheduler)
    else:
        scheduler = None

    monitor = Monitor(
        ("test", partial(evaluate, model, *tst_set)),
        printer=printer,
    )
    for epoch in range(1, opts.epochs + 1):
        for data, target in trn_loader:
            optimizer.zero_grad()
            F.cross_entropy(model(data.to(device)), target.to(device)).backward()
            optimizer.step()
        if epoch % 2 == 0:
            monitor(values={"epoch": epoch, "snorm": model.get_spectral_norms()})
        if scheduler is not None:
            scheduler.step()

    torch.save(monitor.trace, pathlib.Path(opts.out_dir).joinpath("trace.th"))


def main():
    import os
    from liftoff import parse_opts

    opts = parse_opts()

    use_tmp = not (hasattr(opts, "do_not_use_tmp") and opts.do_not_use_tmp)

    if use_tmp and "TMPDIR" in os.environ:
        tmp_path = pathlib.Path(os.environ["TMPDIR"])
        assert os.path.isdir(tmp_path), f"{tmp_path} does not exist; waciuduin?"
        print(f"Will use temporary storage at {tmp_path}")
        opts.tmp_path = tmp_path
    run(opts)


if __name__ == "__main__":
    main()
