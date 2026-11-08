"""Double-DQN agent in pure NumPy (manual back-propagation + Adam).

Fallback used automatically when PyTorch is not installed.  It implements exactly the same
algorithm, network shape (obs -> 128 -> 128 -> actions) and hyper-parameters as
`dqn_torch.py`.
"""
import numpy as np

from .replay import ReplayBuffer


class _MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2.0 / i), (i, o)).astype(np.float32)
                  for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o, np.float32) for o in sizes[1:]]

    def forward(self, x, keep=False):
        acts = [x]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if k < len(self.W) - 1:
                x = np.maximum(x, 0.0)
            acts.append(x)
        return (x, acts) if keep else x

    def backward(self, acts, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for k in reversed(range(len(self.W))):
            gW[k] = acts[k].T @ d
            gb[k] = d.sum(0)
            if k > 0:
                d = (d @ self.W[k].T) * (acts[k] > 0)
        return gW, gb

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


class NumpyDQNAgent:
    kind = "numpy"

    def __init__(self, obs_dim, n_actions, hp, seed=0):
        self.hp, self.n_actions = hp, n_actions
        self.rng = np.random.default_rng(seed)
        sizes = [obs_dim, hp.hidden, hp.hidden, n_actions]
        self.q = _MLP(sizes, self.rng)
        self.target = _MLP(sizes, self.rng)
        self.target.copy_from(self.q)
        self.buffer = ReplayBuffer(hp.buffer_size, obs_dim, seed)
        self.learn_steps = 0
        # Adam state
        self.m = [np.zeros_like(p) for p in self.q.W + self.q.b]
        self.v = [np.zeros_like(p) for p in self.q.W + self.q.b]
        self.t_adam = 0

    def act(self, obs, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q.forward(obs[None, :])[0]))

    def store(self, s, a, r, s2, d):
        self.buffer.add(s, a, r, s2, d)

    def learn(self):
        hp = self.hp
        if len(self.buffer) < max(hp.learn_start, hp.batch_size):
            return None
        s, a, r, s2, d = self.buffer.sample(hp.batch_size)
        n = len(s)
        best = np.argmax(self.q.forward(s2), axis=1)                 # Double DQN
        q_next = self.target.forward(s2)[np.arange(n), best]
        y = r + hp.gamma * (1.0 - d) * q_next
        out, acts = self.q.forward(s, keep=True)
        q_sa = out[np.arange(n), a]
        err = q_sa - y
        huber = np.where(np.abs(err) <= 1.0, 0.5 * err ** 2, np.abs(err) - 0.5)
        dq = np.clip(err, -1.0, 1.0) / n                              # d(Huber)/d(q_sa)
        dout = np.zeros_like(out)
        dout[np.arange(n), a] = dq
        gW, gb = self.q.backward(acts, dout)
        self._adam(gW + gb)
        self.learn_steps += 1
        if self.learn_steps % hp.target_update == 0:
            self.target.copy_from(self.q)
        return float(huber.mean())

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t_adam += 1
        params = self.q.W + self.q.b
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t_adam)
            vh = self.v[i] / (1 - b2 ** self.t_adam)
            p -= self.hp.lr * mh / (np.sqrt(vh) + eps)   # in-place update of weights

    def save(self, path):
        np.savez(str(path) + ".npz", **{f"W{i}": w for i, w in enumerate(self.q.W)},
                 **{f"b{i}": b for i, b in enumerate(self.q.b)})

    def load(self, path):
        z = np.load(str(path) + ".npz")
        n = len(self.q.W)
        self.q.W = [z[f"W{i}"] for i in range(n)]
        self.q.b = [z[f"b{i}"] for i in range(n)]
        self.target.copy_from(self.q)
