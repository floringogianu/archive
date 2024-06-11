""" Some utils. """
import torch
import torch.nn.functional as F


# def idx2xy(idxs, k, s, d):
#     """ Compute normalized coordinates of patch indices in the original
#         feature space.

#     Args:
#         idxs (torch.tesor): List of **patch** (aka window) indices.
#         k (int): size of the patch (size of window).
#         s (padding): stride of the sliding window.
#         d (int): dimension of the input space on which the sliding window was
#                 computed.

#     Returns:
#         torch.tensor: Normalized (w.r.t. input space) xy coordinates.
#     """
#     radius = k / 2
#     maps = (d - k) / s + 1
#     xys = []
#     for idx in idxs:
#         x, y = idx // maps, idx % maps
#         x_, y_ = x * s + radius, y * s + radius
#         xys.append(torch.tensor([x_ / d, y_ / d]))
#     return torch.stack(xys)


def idx2xy(idxs, k, s, d):
    """ Compute normalized coordinates of patch indices in the original
        feature space.
    """
    radius = k / 2
    maps = (d - k) / s + 1
    # N*T, topk
    x, y = idxs // maps, idxs % maps  # unflatten indices
    # N*T, topk, 1
    x_, y_ = (x * s + radius) / d, (y * s + radius) / d  # patch center
    # N*T, topk, 2
    return torch.cat([x_.unsqueeze(-1), y_.unsqueeze(-1)], dim=-1)


def main():
    """ Entry point. """
    K, S = 7, 4
    N, T, C, W, H = 2, 3, 1, 24, 24

    batch = torch.arange(N * T)[:, None, None, None].expand(N * T, C, W, H)
    batch = batch.view(N, T, C, W, H).float()
    batch = F.unfold(batch.view(-1, C, W, H), kernel_size=K, stride=S)
    batch = batch.permute(0, 2, 1)

    # idxs = torch.randint(0, batch.shape[1], (10,))
    # print(batch.shape, idxs)
    # xy = idx2xy(idxs, K, S, W)

    xy = idx2xy(torch.arange(25), K, S, W)
    print(xy)

    xy = idx2xy(torch.arange(4), 4, 2, 6)
    print(xy)


if __name__ == "__main__":
    main()
