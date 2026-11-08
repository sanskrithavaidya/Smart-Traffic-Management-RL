"""Agent factory: PyTorch DQN when available, otherwise the NumPy twin."""


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def make_agent(obs_dim, n_actions, hp, seed=0, kind="auto"):
    if kind == "auto":
        kind = "torch" if torch_available() else "numpy"
    if kind == "torch":
        from .dqn_torch import TorchDQNAgent
        return TorchDQNAgent(obs_dim, n_actions, hp, seed)
    from .dqn_numpy import NumpyDQNAgent
    return NumpyDQNAgent(obs_dim, n_actions, hp, seed)
