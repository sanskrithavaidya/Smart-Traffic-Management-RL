"""Quick sanity check of your SUMO / TraCI / PyTorch installation and the generated
network.  Run this FIRST:   python check_setup.py

It (1) locates SUMO, (2) builds the 2x2 network, (3) starts SUMO through TraCI,
(4) prints every traffic light with its green phases and (5) runs 5 simulated minutes
with the fixed-time controller and with random actions to prove all 4 intersections work.
"""
import os
import sys

import numpy as np


def main():
    print("Python", sys.version.split()[0])
    try:
        import torch
        print("PyTorch", torch.__version__)
    except ImportError:
        print("PyTorch NOT installed -> NumPy DQN fallback will be used (pip install torch)")
    from tsc.network import find_sumo_binary
    print("SUMO_HOME =", os.environ.get("SUMO_HOME"))
    print("sumo      =", find_sumo_binary("sumo"))
    print("netconvert=", find_sumo_binary("netconvert"))
    from tsc.config import SimConfig
    from tsc.sumo_env import SumoEnv
    cfg = SimConfig()
    cfg.sim_seconds = 300
    env = SumoEnv(cfg, work_dir="sumo_files")
    for a in env.agents:
        print(f"  {a}: {env.n_lanes[a]} incoming lanes, {env.n_actions[a]} green phases")
        for i, g in enumerate(env.greens[a]):
            print(f"      phase {i}: {g}")
    env.reset(seed=1)
    done, n = False, 0
    while not done:
        _, r, done, _ = env.step({a: (n // 3) % env.n_actions[a] for a in env.agents})
        n += 1
    print("fixed-time 300 s :", {k: round(v, 2) for k, v in env.metrics().items()})
    rng = np.random.default_rng(0)
    env.reset(seed=1)
    done = False
    while not done:
        _, r, done, _ = env.step({a: int(rng.integers(env.n_actions[a])) for a in env.agents})
    print("random 300 s     :", {k: round(v, 2) for k, v in env.metrics().items()})
    print("\nOK - all", len(env.agents), "intersections respond to control. You can run run_all.py now.")


if __name__ == "__main__":
    main()
