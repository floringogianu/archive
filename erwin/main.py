""" Entry point.
"""
import rlog
import torch
from liftoff import parse_opts
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import src.io_utils as U
from src.data_factories import get_dsets, get_unaugmented
from src.schrodinger import SVIModel


class CifarConvNet(nn.Module):
    """ CNN for CIFAR10 """

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.maxp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxp2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(500, hidden_dim)
        self.out = nn.Linear(hidden_dim, 10)
        self.out_activ = nn.LogSoftmax(1)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return self.out_activ(x)


class SVILoss(nn.Module):
    """ This is actually an ELBO and is of the form
        `NLL_loss + KL(q(phi | p(theta))` where `phi` are the variational
        parameters and `theta` the model's parameters.
    """

    def __init__(self, kl_div, nll_weight=None, reduction="mean"):
        super(SVILoss, self).__init__()
        self.kl_div = kl_div
        self.nll_weight = nll_weight
        self.reduction = reduction

    def forward(self, output, target):  # pylint: disable=arguments-differ
        # compute the NLL
        nll = F.nll_loss(output, target, reduction=self.reduction)
        if self.nll_weight is not None:
            nll *= self.nll_weight
        # and the kl
        kl = self.kl_div()
        return nll + kl


def train(loader, model, optimizer, criterion, mc_samples=0):
    """ Training routine.

    Args:
        loader (torch.utils.data.DataLoader): The data.
        model (schrodinger.VariationalModel): A variational wrapper over a
            user-defined MLE model.
        data_loader (torch.utils.data.DataLoader): Reads and prepares data.
        optimizer (torch.optim.Optimizer): The stochastic gradient descent
            optimization method.
        criterion: (torch.nn.Module): A loss function.
    """

    device = list(model.parameters())[0].device
    total_loss, correct = 0, 0

    model.train()
    for _, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if mc_samples:
            # get the model predictions by marginalizatin
            output = model.forward(data, M=mc_samples)
        else:
            # or MLE
            output = model.forward(data)

        # compute the loss function. For SVI
        loss = criterion(output, target)

        loss.backward()

        if mc_samples is not None:
            # TODO: what was this?
            model._normals = None
        optimizer.step()

        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()

    total_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return total_loss, accuracy


def validate(loader, model, mc_samples=0):
    """ Validation routine.
    """
    device = list(model.parameters())[0].device
    nll_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            if mc_samples:
                # get the model's prediction using M samples from the posterior
                output = model.forward(data, M=mc_samples)
            else:
                output = model.forward(data)

            nll_loss += F.nll_loss(output, target, reduction="sum").item()

            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # TODO: forgot what is this about
    if mc_samples:
        model._normals = None

    nll_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return nll_loss, accuracy


def get_model(opt, device):
    """ Return a suitable model. """
    model = CifarConvNet(hidden_dim=512).to(device)
    if opt.mode == "SVI":
        return SVIModel(model)
    return model


def get_criterion(opt, model, nll_weight):
    """ Return a suitable loss function. """
    if opt.mode == "SVI":
        rlog.info("\nLoss: NLL + KL")
        return SVILoss(model.get_kl_div, nll_weight=nll_weight)
    rlog.info("Loss: NLL \n")
    return nn.NLLLoss()


def train_stats(opt, model, dset):
    """ Stats on the traning data.
    """
    stats = {}
    if isinstance(model, SVIModel):
        # Stats collected during traning use a single sample from the
        # posterior. Therefore we check the accuracy once more using
        # the same no of samples as on the validation set.
        stats["lossMC"], stats["accMC"] = validate(
            DataLoader(dset, **vars(opt.val_loader)), model, opt.tst_mcs
        )
    if hasattr(opt, "log") and opt.log.train_no_aug:
        # We also look at the accuracy on un-augmented training data.
        # This is done on both MLE and SVI
        rlog.info("Compute accuracy on un-augmented train data.")
        mc_samples = opt.tst_mcs if isinstance(model, SVIModel) else 0
        stats["lossNoAug"], stats["accNoAug"] = validate(
            DataLoader(get_unaugmented(dset), **vars(opt.val_loader)),
            model,
            mc_samples,
        )
    if hasattr(opt, "log") and opt.log.mle_ish:
        # Use the means of the posterior to set a pseudo-MLE model.
        assert isinstance(
            model, SVIModel
        ), "This stat only makes sense for SVI models."
        model.sync_mle_model()
        rlog.info("Synced MLE model using means from posterior.")
        rlog.info("Compute accuracy with a pseudo-MLE model.")
        stats["lossMLE"], stats["accMLE"] = validate(
            DataLoader(dset, **vars(opt.val_loader)),
            model._mle_model,  # pylint: disable=protected-access
            0,
        )
    return stats


def valid_stats(opt, model, dset):
    """ Stats on the validation data.
    """
    stats = {}
    stats["loss"], stats["acc"] = validate(
        DataLoader(dset, **vars(opt.val_loader)), model, opt.tst_mcs
    )

    if hasattr(opt, "log") and opt.log.mle_ish:
        # Use the means of the posterior to set a pseudo-MLE model.
        assert isinstance(
            model, SVIModel
        ), "This stat only makes sense for SVI models."
        model.sync_mle_model()
        rlog.info("Synced MLE model using means from posterior.")
        rlog.info("Compute accuracy with a pseudo-MLE model.")
        stats["lossMLE"], stats["accMLE"] = validate(
            DataLoader(dset, **vars(opt.val_loader)),
            model._mle_model,  # pylint: disable=protected-access
            0,
        )
    return stats


def model_stats(opt, epoch, model):
    """ Log additional model stats.
    """
    log = rlog.getLogger(opt.experiment + ".model")
    if hasattr(opt, "log") and opt.log.detailed:
        # log histogram also
        assert isinstance(
            model, SVIModel
        ), "This stat only makes sense for SVI models."
        for mu, std in zip(model.mu(), model.std()):
            log.put(mu=mu, std=std)
        log.trace(step=epoch, **model.summarize())
        log.reset()


def set_logger(opt):
    """ Configure logger.
    """
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
    trn_log = rlog.getLogger(opt.experiment + ".train")
    val_log = rlog.getLogger(opt.experiment + ".valid")
    trn_log.fmt = "[{:03d}][TRN]  acc={:5.2f}%  loss={:5.2f}"
    val_log.fmt = "[{:03d}][VAL]  acc={:5.2f}%  loss={:5.2f}"
    # add histogram support
    if hasattr(opt, "log") and opt.log.detailed:
        mdl_log = rlog.getLogger(opt.experiment + ".model")
        mdl_log.addMetrics(
            rlog.ValueMetric("std", metargs=["std"], tb_type="histogram"),
            rlog.ValueMetric("mu", metargs=["mu"], tb_type="histogram"),
        )
    return trn_log, val_log


def run(opt):
    """ Run experiment. This function is being launched by liftoff.
    """
    # logging
    trn_log, val_log = set_logger(opt)

    # model related stuff
    device = torch.device("cuda")
    trn_set, val_set, wmp_set = get_dsets(opt)
    model = get_model(opt, device)
    optimizer = getattr(optim, opt.optim.name)(
        model.parameters(), **vars(opt.optim.args)
    )
    # batch_size
    batch_size = opt.trn_loader.batch_size

    rlog.info(U.config_to_string(opt))
    rlog.info("Model: %s", str(model))
    rlog.info("Optimizer: %s \n", str(optimizer))

    # Warm-up the mode on a partition of the training dataset
    if wmp_set is not None:
        rlog.info("Warming-up on dset of size %d", len(wmp_set))
        for epoch in range(opt.warmup.epochs):
            # train for one epoch
            trn_loss, trn_acc = train(
                DataLoader(wmp_set, **vars(opt.trn_loader)),
                model,
                optimizer,
                get_criterion(opt, model, len(wmp_set) // batch_size),
                mc_samples=opt.trn_mcs,
            )

            val_stats = valid_stats(opt, model, val_set)
            trn_stats = train_stats(opt, model, wmp_set)
            trn_stats["loss"], trn_stats["acc"] = trn_loss, trn_acc

            # to pickle and tensorboard
            val_log.trace(step=epoch, **val_stats)
            trn_log.trace(step=epoch, **trn_stats)

            # to console
            for log, stats in zip([trn_log, val_log], [trn_stats, val_stats]):
                log.info(log.fmt.format(epoch, stats["acc"], stats["loss"]))

            # extra logging
            model_stats(opt, epoch, model)

        # maybe reset optimizer after warmup
        if opt.warmup.reset_optim:
            rlog.info("\nWarmup ended. Resetting optimizer.")
            optimizer = getattr(optim, opt.optim.name)(
                model.parameters(), **vars(opt.optim.args)
            )

    # Train on the full training dataset
    if wmp_set is not None:
        epochs = range(opt.warmup.epochs, opt.warmup.epochs + opt.epochs)
    else:
        epochs = range(opt.epochs)

    rlog.info("\nTraining on dset: %s", str(trn_set))
    for epoch in epochs:
        trn_loss, trn_acc = train(
            DataLoader(trn_set, **vars(opt.trn_loader)),
            model,
            optimizer,
            get_criterion(opt, model, len(trn_set) // batch_size),
            mc_samples=opt.trn_mcs,
        )

        val_stats = valid_stats(opt, model, val_set)
        trn_stats = train_stats(opt, model, trn_set)
        trn_stats["loss"], trn_stats["acc"] = trn_loss, trn_acc

        # to pickle and tensorboard
        val_log.trace(step=epoch, **val_stats)
        trn_log.trace(step=epoch, **trn_stats)

        # to console
        for log, stats in zip([trn_log, val_log], [trn_stats, val_stats]):
            log.info(log.fmt.format(epoch, stats["acc"], stats["loss"]))

        # extra logging
        model_stats(opt, epoch, model)


def main():
    """ Read config file using liftoff and launch experiments. You get here
        only if you launch multiple experiments using liftoff.
    """
    opt = parse_opts()
    run(opt)


if __name__ == "__main__":
    main()
