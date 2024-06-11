import torch
import numpy as np


class RolloutBufferNormalDist:
    def __init__(self):
        self.a_idx_memory = []
        self.pi_memory = []
        self.r_memory = []
        self.v_memory = []
        self.entropy_memory = []
        self.t = 0

    def push(
        self,
        v: torch.tensor,
        r: np.array,
        a_idx: torch.tensor,
        pi: torch.tensor,
        entropy: np.array,
    ):
        self.v_memory.append(v)
        self.r_memory.append(r)
        self.a_idx_memory.append(a_idx)
        self.pi_memory.append(pi)
        self.entropy_memory.append(entropy)

    def get_data(self) -> list:
        v = self.v_memory
        r = self.r_memory
        a_idx = self.a_idx_memory
        pi = self.pi_memory
        entropy = self.entropy_memory
        self.v_memory = []
        self.r_memory = []
        self.a_idx_memory = []
        self.pi_memory = []
        self.entropy_memory = []
        return v, r, a_idx, pi, entropy
