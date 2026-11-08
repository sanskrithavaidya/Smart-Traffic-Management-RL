"""Double-DQN agent implemented in PyTorch (primary implementation)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay import ReplayBuffer


class QNetwork(nn.Module):
    """MLP: obs -> 128 -> 128 -> Q-value per action."""

    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class TorchDQNAgent:
    """One independent learner (own network, target network and replay buffer)."""
    kind = "torch"

    def __init__(self, obs_dim, n_actions, hp, seed=0):
        torch.manual_seed(seed)
        self.hp, self.n_actions = hp, n_actions
        self.rng = np.random.default_rng(seed)
        self.q = QNetwork(obs_dim, n_actions, hp.hidden)
        self.target = QNetwork(obs_dim, n_actions, hp.hidden)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=hp.lr)
        self.buffer = ReplayBuffer(hp.buffer_size, obs_dim, seed)
        self.learn_steps = 0

    @torch.no_grad()
    def act(self, obs, eps=0.0):
        """Epsilon-greedy action."""
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        qv = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(qv.argmax(dim=1).item())

    def store(self, s, a, r, s2, d):
        self.buffer.add(s, a, r, s2, d)

    def learn(self):
        hp = self.hp
        if len(self.buffer) < max(hp.learn_start, hp.batch_size):
            return None
        s, a, r, s2, d = [torch.as_tensor(x) for x in self.buffer.sample(hp.batch_size)]
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():                       # Double DQN target
            best = self.q(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, best).squeeze(1)
            y = r + hp.gamma * (1.0 - d) * q_next
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.learn_steps += 1
        if self.learn_steps % hp.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.q.state_dict(), str(path) + ".pt")

    def load(self, path):
        self.q.load_state_dict(torch.load(str(path) + ".pt", map_location="cpu"))
        self.target.load_state_dict(self.q.state_dict())
