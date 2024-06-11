from itertools import repeat
from pathlib import Path

import rlog
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from liftoff import parse_opts
from torchvision.utils import save_image

import src.io_utils as ioutil
from lib import load_dataset


class VAE(nn.Module):
    def __init__(self, in_channels=1, zdim=10):
        super(VAE, self).__init__()

        self.zdim = zdim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.Flatten(),
            nn.Linear(256, 2 * zdim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4 * 4 * 64),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        mu, logvar = torch.split(self.encoder(x), self.zdim, -1)
        std = torch.exp(0.5 * logvar)
        q = D.Normal(mu, std)
        z = q.rsample()
        return self.decoder(z), q


class DoVAE(VAE):
    def __init__(self, in_channels=1, zdim=10):
        super().__init__(in_channels=in_channels, zdim=zdim)

    def do(self, x):
        mu, logvar = torch.split(self.encoder(x), self.zdim, -1)
        std = torch.exp(0.5 * logvar)
        q = D.Normal(mu, std)
        z = q.rsample()

        # we need more ways of setting zi
        B = x.size(0)
        do_z = [z.clone() for _ in range(self.zdim)]
        for i, z_k in enumerate(do_z):
            z_k[:, i] = z_k[torch.randint(0, B, (1,)), i]
        do_z = torch.cat(do_z, dim=0)
        return self.decoder(do_z)
        # return torch.split(self.decoder(z), B, 0)


class Sampler:
    def __init__(self, path, model, loaders, N=64) -> None:
        self.path = Path(path) / "samples"
        self.model = model
        self.device = next(model.parameters()).device
        self.N = N

        self.path.mkdir(parents=True, exist_ok=True)
        self._set_recon_groundtruths(loaders)

    def __call__(self, step, sample_prior=False):
        self.check_reconstruction(step)
        if sample_prior:
            self.generate_samples(step)

    def check_reconstruction(self, step):
        for name, xs in self.recon_samples.items():
            with torch.no_grad():
                xs_, _ = self.model(xs.to(self.device))
                xs_ = torch.clip(xs_, 0.0, 1.0).cpu()
                save_image(xs_, self.path / f"{name}_recon_{step:06d}.png")

    def generate_samples(self, step):
        with torch.no_grad():
            z = torch.randn((self.N, self.model.zdim)).to(self.device)
            xs_ = self.model.decoder(z).clip_(0, 1).cpu()
            save_image(xs_, self.path / f"sample_{step:06d}.png")

    def _set_recon_groundtruths(self, loaders):
        d = {}
        for name, loader in loaders.items():
            dset = loader.dataset
            idxs = torch.randperm(len(dset))[: self.N]
            xs = torch.stack([dset[i][0] for i in idxs])
            save_image(xs, self.path / f"{name}_gt.png")
            d[name] = xs
        self.recon_samples = d


def infinite_loader(data_loader):
    for loader in repeat(data_loader):
        for batch in loader:
            yield batch


def validate(model, loader, log):
    device = next(model.parameters()).device
    for (x, _) in loader:
        x = x.to(device)
        with torch.no_grad():
            x_, q = model(x)
            _, info = loss_fn(x_, x, q, beta=1)
        log.put(**info)


MODELS = {"VAE": VAE, "DoVAE": DoVAE}


def loss_fn(x_, x, q, beta=1.0, info=True, reduce=True):
    B = x.size(0)

    # compute nll
    if x.size(1) == 1:
        nll = F.binary_cross_entropy_with_logits(x_, x, reduction="none")
    else:
        nll = F.mse_loss(x_, x, reduction="none")

    # compute KL divergence
    prior = D.Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))
    kld = D.kl_divergence(q, prior)

    elbo = nll.view(B, -1).sum(-1, keepdim=True) + (beta * kld).sum(-1, keepdim=True)

    if reduce:
        elbo = elbo.sum()

    if info:
        return (
            elbo,
            {
                k: v.detach().sum().cpu().item()
                for k, v in zip(["elbo", "nll", "kld"], [elbo, nll, kld])
            },
        )
    return elbo


def run(opt):
    opt.device = torch.device(opt.device)

    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)
    trn_log = rlog.getLogger(f"{opt.experiment}.train")
    val_log = rlog.getLogger(f"{opt.experiment}.valid")
    B = opt.batch_size
    trn_log.addMetrics(
        rlog.AvgMetric("nll", metargs=["nll", B]),
        rlog.AvgMetric("kld", metargs=["kld", B]),
        rlog.AvgMetric("elbo", metargs=["elbo", B]),
        rlog.AvgMetric("loss", metargs=["elbo", 1]),
        rlog.AvgMetric("REx", metargs=["rex", 1]),
    )
    val_log.addMetrics(
        rlog.AvgMetric("nll", metargs=["nll", B]),
        rlog.AvgMetric("kld", metargs=["kld", B]),
        rlog.AvgMetric("elbo", metargs=["elbo", B]),
        rlog.AvgMetric("loss", metargs=["elbo", 1]),
    )

    trn_ldr = load_dataset(
        opt.dset_name,
        variant=opt.trn_variant,
        mode="train",
        dataset_path="./data/",
        batch_size=opt.batch_size,
        num_workers=4,
    )
    val_ldr = load_dataset(
        opt.dset_name,
        variant=opt.val_variant,
        mode="train",  # we validate on training data
        dataset_path="./data/",
        batch_size=opt.batch_size,
        num_workers=4,
    )
    opt.model.args["in_channels"] = trn_ldr.dataset[0][0].shape[0]

    model = MODELS[opt.model.name](**opt.model.args).to(opt.device)
    optim = getattr(O, opt.optim.name)(model.parameters(), **opt.optim.args)
    sampler = Sampler(
        opt.out_dir, model, {opt.trn_variant: trn_ldr, opt.val_variant: val_ldr}
    )

    rlog.info(ioutil.config_to_string(opt))

    trn_loader = infinite_loader(trn_ldr)
    for step in range(1, opt.trn_steps + 1):

        x, _ = next(trn_loader)
        x = x.to(opt.device)

        x_, q = model(x)
        loss, info = loss_fn(x_, x, q, beta=opt.beta)

        if opt.model.name == "DoVAE":
            with torch.no_grad():
                x_envs = model.do(x)

            x_envs_rec, q_envs = model(x_envs)
            risks = loss_fn(x_envs_rec, x_envs, q_envs, reduce=False, info=False)

            risks = risks.view(model.zdim, opt.batch_size)  # Z * B
            risks = risks.sum(-1, keepdim=True)  # sum the errors in each "env"
            risks_var = risks.var()

            loss += opt.rex_coeff * risks_var
            info["rex"] = risks_var.detach().cpu().item()

        model.zero_grad()
        loss.backward()
        optim.step()

        trn_log.put(**info)
        if step % opt.val_freq == 0:
            sampler(step, sample_prior=True)
            validate(model, val_ldr, val_log)

            trn_log.traceAndLog(step)
            val_log.traceAndLog(step)


if __name__ == "__main__":
    run(parse_opts())

