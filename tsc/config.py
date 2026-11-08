"""Central configuration: grid geometry, simulation, reward, DQN and training settings."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- network layout
# 2 x 2 grid of signalised intersections.  "J<row><col>", row 0 is the top row.
JUNCTIONS = ["J00", "J01", "J10", "J11"]
DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}  # (d_row, d_col)
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}  # heading -> heading after a left turn


def junction_rc(j):
    """'J10' -> (1, 0)"""
    return int(j[1]), int(j[2])


def neighbour_map():
    """Adjacent signalised junctions (used for agent coordination)."""
    out = {j: [] for j in JUNCTIONS}
    for a in JUNCTIONS:
        ra, ca = junction_rc(a)
        for b in JUNCTIONS:
            rb, cb = junction_rc(b)
            if abs(ra - rb) + abs(ca - cb) == 1:
                out[a].append(b)
    return out


NEIGHBOURS = neighbour_map()

# ---------------------------------------------------------------- demand
# Entry rate (veh/h) per border entry, depending on the side of the grid it sits on.
ENTRY_RATE_VPH = {"N": 300.0, "S": 300.0, "E": 450.0, "W": 450.0}
# Time-varying demand multiplier: (begin_s, end_s, factor) -> morning-peak style profile.
DEMAND_PROFILE = [(0, 1200, 0.8), (1200, 2400, 1.4), (2400, 3600, 0.8)]


@dataclass
class SimConfig:
    sim_seconds: int = 3600      # one episode = one simulated hour
    delta_time: int = 10         # seconds between two agent decisions
    yellow_time: int = 3         # yellow interval when a phase is changed
    link_length: float = 250.0   # metres between intersections
    free_speed: float = 13.89    # m/s (50 km/h)
    lanes_per_edge: int = 2
    veh_space: float = 7.5       # metres per queued vehicle (length + min gap)
    # reward  r = -(w_queue * halting_vehicles + w_wait * sum_of_waiting_seconds)
    reward_w_queue: float = 0.1
    reward_w_wait: float = 1.0 / 300.0
    reward_floor: float = -10.0
    fixed_green: int = 30        # green time (s) per phase for the fixed-time baseline


@dataclass
class DQNConfig:
    hidden: int = 128
    lr: float = 5e-4
    gamma: float = 0.9
    batch_size: int = 64
    buffer_size: int = 50000
    target_update: int = 200     # learning steps between hard target-network copies
    learn_start: int = 500       # transitions before learning starts
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7  # fraction of training episodes over which eps decays
    coop_alpha: float = 0.3      # weight of neighbours' reward in each agent's training reward


@dataclass
class TrainConfig:
    episodes: int = 150
    eval_every: int = 10         # validation every N training episodes
    val_seeds: tuple = (900, 901, 902)
    test_seeds: tuple = tuple(range(1000, 1010))
    train_seed0: int = 1
    train_scale_range: tuple = (0.8, 1.25)  # random demand jitter during training
    test_scales: tuple = (1.0,)
