import pickle
from pathlib import Path

import lz4.frame
import numpy as np
import torch
from rich import print, progress
from torchvision.utils import make_grid, save_image

from src.replay import OfflineExperienceReplay, load_file


def main():
    for ridx in range(51):
        split_one_replay(ridx, game="Seaquest")
    # with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    #     executor.map(partial(split_one_replay, game="Riverraid"), range(5))


def load_checkpoint(root, idx):
    """ Loads all the checkpoint files for a given checkpoint index. """
    fpaths = []
    fpaths = [
        root / "$store$_{}_ckpt.{}.gz".format(el, idx)
        for el in ["observation", "action", "reward", "terminal"]
    ]
    fpaths += [
        root / "{}_ckpt.{}.gz".format(el, idx) for el in ["add_count", "invalid_range"]
    ]
    objects = [load_file(fpath) for fpath in fpaths]
    return objects[:4], objects[4:]


def split_one_replay(ridx, game="Seaquest", seed="1", step=50):
    src_root = Path(f"/atari/{game}/{seed}/replay_logs")
    dst_root = Path("/data/datasets/atari") / game.lower() / seed / f"r{ridx:02}"
    dst_root.mkdir(parents=True)

    data, (_, invalid) = load_checkpoint(src_root, ridx)

    msg = f"[green]Replay={ridx:02d}..."
    for idx in progress.track(range(0, len(data[0]) - step + 1, step), description=msg):
        transition = [el[idx : idx + step] for el in data]
        transition.append(np.array([i not in invalid for i in range(idx, idx + step)]))
        # compress and write the transition
        fpath = dst_root / "r{:02d}_t{:06d}.lz4".format(ridx, idx)
        with lz4.frame.open(fpath, "w") as payload:
            payload.write(pickle.dumps(transition))


def split_one_replay_(replay_idx, game="Seaquest", seed="1"):
    replay = OfflineExperienceReplay(
        f"/atari/{game}/{seed}/replay_logs",
        True,
        capacity=1000000,
        device="cpu",
        batch_size=1,
    )

    replay.reload_N(N=1, ckpt_idxs=[replay_idx])

    msg = f"[green]Replay={replay_idx:02d}..."
    for idx in progress.track(range(3, 999998), description=msg):
        transition = replay.sample(idxs=[idx])

        fpath = "/data/datasets/atari/{}/{}/r{:02d}_t{:06d}.lz4".format(
            game.lower(), seed, replay_idx, idx
        )
        with lz4.frame.open(fpath, "w") as payload:
            payload.write(pickle.dumps(transition))


def invalids():
    for idx in range(50):
        fp = f"/atari/Seaquest/1/replay_logs/invalid_range_ckpt.{idx}.gz"
        print(idx, load_file(fp))


def show(replay):
    samples = []
    for idx in range(230747, 230753):
        state, action, reward, state_, done = replay.sample(idxs=[idx])
        samples.append(state.swapaxes(0, 1))
        print(action)
    samples = torch.cat(samples)
    print(samples.shape, samples.max())
    img = make_grid(samples, nrow=4).float() / 255
    save_image(img, "./samples_replay.png")


if __name__ == "__main__":
    # main()
    replay = OfflineExperienceReplay(
        f"/atari/Seaquest/1/replay_logs",
        True,
        capacity=1000000,
        device="cpu",
        batch_size=1,
    )
    replay.reload_N(ckpt_idxs=[0])

    show(replay)
