import torch
from torch.distributions import Normal


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Intensity:
    def __init__(self, mu=1.0, sigma=0.1, p=0.5):
        self.alpha = Normal(mu, sigma)
        self.p = p

    def __call__(self, x):
        if torch.rand(1) > self.p:
            return x
        s = self.alpha.sample().clamp_(0.8, 1.2).to(x.device)
        x = (x.float() * s).clamp_(0, 255)
        return x.byte()


class HorizontalShift:
    def __init__(self, padding=4):
        self.pad = padding
        # avoid allocating memory
        self.pad_buff = None
        self.result = None

    def _hpad(self, x):
        N, C, H, W = x.shape
        x = x.view(N * C * H, W)

        if self.pad_buff is None:
            self.pad_buff = torch.zeros(
                N * C * H, W + self.pad * 2, dtype=x.dtype, device=x.device
            )

        for col in range(self.pad):
            self.pad_buff[:, col] = x[:, 0]  # pad with the left margin
        for col in range(W, W + self.pad):
            self.pad_buff[:, col] = x[:, -1]  # pad with the right margin

        # add the rest of the image in the center
        self.pad_buff[:, self.pad : W + self.pad] = x

        # print(self.pad_buff[:, 0])

        return self.pad_buff.view(N, C, H, W + self.pad * 2)

    def __call__(self, x):
        """ Assumes a B * 4 * 84 * 84 input. """
        N, C, H, W = x.shape
        if self.result is None:
            self.result = torch.zeros_like(x)

        xpad = self._hpad(x)

        shifts = []
        for i in range(N):
            start_idx = torch.randint(0, self.pad * 2, (1,)).item()
            end_idx = start_idx + W
            shifts.append(xpad[i, :, :, start_idx:end_idx])

        torch.stack(shifts, out=self.result)
        return self.result

