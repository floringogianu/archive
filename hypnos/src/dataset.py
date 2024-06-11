import pickle
from pathlib import Path

import lz4.frame
import numpy as np
import torch
from rich.progress import track
from torch.utils.data import Dataset
from torchvision.utils import make_grid, save_image


def load_transitions_chunk(fpath):
    with lz4.frame.open(fpath, mode="r") as f:
        data = pickle.loads(f.read())
    return data


def _get_file_paths(root, total):
    file_paths = []
    for fp in track(Path(root).glob("**/*.lz4"), total=total):
        file_paths.append(fp)
    return sorted(file_paths, key=lambda f: f.stem)


class Atari(Dataset):
    def __init__(self, root, split="train", chunk_size=50, histlen=4):
        self.__file_paths = _get_file_paths(root, int(50 * 1e6 / chunk_size))
        self.__file_paths = self.__file_paths
        self.__chunk_size = chunk_size
        self.__histlen = histlen

    def __getitem__(self, idx):
        idx = idx % 5000
        cidx = idx // self.__chunk_size
        tidx = idx % self.__chunk_size

        # chunk is (state, action, reward, done, is_valid)
        chunk = load_transitions_chunk(self.__file_paths[cidx])

        # check if sampled transition is valid or if is terminal
        # and resample if it's not. That's right, we don't train on final
        # transitions because the very last state is usually not
        # present in the Dopamine replay buffer.
        done, valid = chunk[-2][tidx], chunk[-1][tidx]
        if (not valid) or done:
            while (not valid) or done:
                tidx = np.random.randint(self.__chunk_size)
                done, valid = chunk[-2][tidx], chunk[-1][tidx]

        # prepend or append another chunk if the tidx is at extremities
        chunk, tidx = self._maybe_load_before_or_after(cidx, tidx, chunk)

        # start composing the sars_ transition
        chunk = list(zip(*chunk))
        s, a, r, d, _ = chunk[tidx]
        s_ = [chunk[tidx + 1][0]]
        s = [s]

        last_screen = s[0]
        found_done = False
        bidx = tidx

        for _ in range(self.__histlen - 1):
            # append the previous observation to the next state
            s_.append(s[-1])
            # append the previous observation to this state.
            # if we iterate to the end of the previous episode we just
            # duplicate the observation
            if not found_done:
                bidx -= 1
                new_transition = chunk[bidx]
                if new_transition[3]:
                    found_done = True
                else:
                    last_screen = new_transition[0]
            s.append(last_screen)
        return np.stack(s[::-1]), a, r, np.stack(s_[::-1]), d

    def _maybe_load_before_or_after(self, cidx, tidx, chunk):
        if tidx in [0, 1, 2]:
            _chunk = load_transitions_chunk(self.__file_paths[cidx - 1])
            chunks = zip(_chunk, chunk)
            tidx += self.__chunk_size
        elif tidx == (self.__chunk_size - 1):
            chunk_ = load_transitions_chunk(self.__file_paths[cidx + 1])
            chunks = zip(chunk, chunk_)
        else:
            return chunk, tidx
        chunk = [np.concatenate((a, b)) for a, b in chunks]
        return chunk, tidx

    def _make_state(self):
        pass

    def __len__(self):
        return len(self.__file_paths) * self.__chunk_size


def show(dset):
    samples = []
    for idx in range(230747, 230753):
        state, action, reward, state_, done = dset[idx]
        state = torch.from_numpy(state).unsqueeze(0)
        state_ = torch.from_numpy(state_).unsqueeze(0)
        samples.append(state.swapaxes(0, 1))
        print(action)
        # samples.append(state_.swapaxes(0, 1))
    samples = torch.cat(samples)
    img = make_grid(samples, nrow=4).float() / 255
    save_image(img, "./samples_loader.png")


def collate(batch):
    states = torch.from_numpy(np.stack([el[0] for el in batch]))
    states_ = torch.from_numpy(np.stack([el[3] for el in batch]))
    actions = torch.tensor([el[1] for el in batch], dtype=torch.long).unsqueeze(0)
    rewards = torch.tensor([el[2] for el in batch], dtype=torch.float32).unsqueeze(0)
    done = torch.tensor([el[4] for el in batch], dtype=torch.bool).unsqueeze(0)
    return states, actions, rewards, states_, done


if __name__ == "__main__":
    dset = Atari("/data/datasets/atari/seaquest/1/")
    # dloader = torch.utils.data.DataLoader(dset, batch_size=32, num_workers=4)
    # for i, (states, actions, rewards, states_, done) in enumerate(dloader):
    #     if i % 1000 == 0:
    #         print(i, states.shape, actions.shape, done.shape)

    show(dset)

    # for idx in range(int(2e6)):
    #     s, a, r, s_, d = dset[idx]
    #     if d:
    #         print(idx, s.shape, s_.shape, a, r, d)
    #     for i in range(3):
    #         assert (s[i] == s_[i + 1]).all(), "Qhooops"
