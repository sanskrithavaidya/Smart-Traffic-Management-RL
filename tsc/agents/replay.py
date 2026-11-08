"""Uniform experience replay buffer (NumPy), shared by both DQN implementations."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, seed=0):
        self.cap = capacity
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.n = 0
        self.i = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d):
        k = self.i
        self.s[k], self.a[k], self.r[k], self.s2[k], self.d[k] = s, a, r, s2, float(d)
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.n
