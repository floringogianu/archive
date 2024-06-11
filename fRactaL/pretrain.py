import rlog
import torch
import torch.nn as nn
import torchvision.transforms as T
from liftoff import parse_opts
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class AtariFractals(ImageFolder):
    def __init__(self, root, split="train"):
        super().__init__(root,)
        self.samples = self._set_split(split)
        if "84" in root:
            self._get_context = self._get_past_context
            print("Using past context with: ", root)
        else:
            # self._get_context = self._get_spread_context
            # print("Using spread context with: ", root)
            self._get_context = self._get_sliding_context
            self._crop_fns = [T.RandomCrop((210, 84)), T.RandomCrop((84, 210))]
            print("Using sliding context with: ", root)
        self._im2th = T.ToTensor()

    def _set_split(self, split):
        generator = torch.Generator().manual_seed(42)  # split rnd generator
        rand_idxs = torch.randperm(len(self), generator=generator)
        split_idx = int(len(self) * 0.9)
        trn_split, val_split = rand_idxs[:split_idx], rand_idxs[split_idx:]
        if split == "train":
            self.samples = [self.samples[i] for i in trn_split]
        elif split == "valid":
            self.samples = [self.samples[i] for i in val_split]
        else:
            raise ValueError(f"Unknown split `{split}`. Use either `train` or `valid`.")
        return sorted(self.samples, key=lambda sample: (sample[1], sample[0]))

    def __getitem__(self, index):
        obs, label = self._get_context(index)
        return obs, label

    def _get_spread_context(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        img = img.convert("L")  # both PIL and ALE use (.299/.587/.114)
        img = self._im2th(img)
        # get some crops
        img = torch.cat(
            [
                c
                for i, c in enumerate(T.functional.five_crop(img, 84))
                if i != index % 5
            ],
            dim=0,
        )
        # img = torch.cat(
        #     [
        #         T.functional.crop(img, x.item(), y.item(), 84, 84)
        #         for x, y in torch.randint(0, (img.shape[-1] - 84), (4, 2))
        #     ],
        #     dim=0,
        # )
        return img, label

    def _get_sliding_context(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        img = img.convert("L")  # both PIL and ALE use (.299/.587/.114)
        img = self._im2th(img)
        # crop verticaly or horizontaly
        crop_fn = self._crop_fns[torch.randint(2, (1,)).item()]
        img = crop_fn(img)  # 84 x 210, allowing for some overlap with a step of 42
        img = img.unfold(1, 84, 42).unfold(2, 84, 42).squeeze()
        return img, label

    def _get_past_context(self, index):
        obs_buff = []
        _, label0 = self.samples[index]

        for t in range(4):
            if index - t < 0:
                img = torch.zeros(1, 84, 84)
            else:
                path, label = self.samples[index - t]
                if label != label0:
                    img = torch.zeros(1, 84, 84)
                else:
                    img = self.loader(path)
                    img = img.convert("L")  # both PIL and ALE use (.299/.587/.114)
                    img = self._im2th(img)
            obs_buff.append(img)
        obs = torch.cat(obs_buff[::-1], 0)
        return obs, label


class AtariNet(nn.Module):
    """ Estimator used by DQN for ATARI games.
    """

    def __init__(self, action_no, fc_layers=(512, 512)):
        super().__init__()

        self.action_no = out_size = action_no

        # get the feature extractor and fully connected layers
        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        head, in_size = [], 64 * 7 * 7
        for i, hidden_size in enumerate(fc_layers):
            if i != len(fc_layers) - 1:
                head += [
                    nn.Linear(in_size, hidden_size),
                    nn.ReLU(inplace=True),
                ]
            else:
                head.append(nn.Linear(in_size, out_size))
            in_size = hidden_size
        self.__head = nn.Sequential(*head)

    def forward(self, x):
        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))
        return qs


def get_dataloaders(load_root, batch_size):
    trn_ldr = DataLoader(
        AtariFractals(load_root, split="train"),
        shuffle=True,
        batch_size=batch_size,
        num_workers=12,
    )
    val_ldr = DataLoader(
        AtariFractals(load_root, split="valid"),
        shuffle=True,
        batch_size=2 * batch_size,
        num_workers=12,
    )
    return trn_ldr, val_ldr


def train(model, optim, loader, device):
    criterion = nn.CrossEntropyLoss()
    for i, (x, t) in enumerate(loader):
        x, t = x.to(device), t.to(device)
        y = model(x)

        optim.zero_grad()
        loss = criterion(y, t)
        loss.backward()
        optim.step()

        _, pred = torch.max(y.data, 1)
        rlog.put(
            trn_loss=loss.detach().item(),
            trn_correct=(pred == t).sum().item(),
            trn_bsz=x.shape[0],
        )

        if i % 200 == 0 and i > 0:
            tmpl = "[{batch:4d}/{left:4d}]  loss={trn_loss:2.4f},  acc={trn_acc:2.4f}"
            rlog.info(tmpl.format(batch=i, left=len(loader), **rlog.summarize()))


def validate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, t in loader:
            x, t = x.to(device), t.to(device)
            y = model(x)

            loss = criterion(y, t)

            _, pred = torch.max(y.data, 1)
            rlog.put(
                val_loss=loss.detach().item(),
                val_correct=(pred == t).sum().item(),
                val_bsz=x.shape[0],
            )


def run(opt):

    device = torch.device("cuda")

    rlog.init("fRactaL", path=opt.out_dir, tensorboard=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.AvgMetric("trn_acc", metargs=["trn_correct", "trn_bsz"]),
        rlog.AvgMetric("val_loss", metargs=["val_loss", 1]),
        rlog.AvgMetric("val_acc", metargs=["val_correct", "val_bsz"]),
    )
    rlog.info(opt)

    model = AtariNet(1000, **opt.model.args)
    model.to(device)
    print(model)
    trn_loader, val_loader = get_dataloaders(opt.load_root, opt.batch_size)
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.epochs):
        train(model, optim, trn_loader, device)
        validate(model, val_loader, device)
        rlog.traceAndLog(epoch)
        torch.save(model.state_dict(), f"{opt.out_dir}/model_{epoch:04d}.pkl")


if __name__ == "__main__":
    run(parse_opts())
