"""End-to-end pipeline:  environment -> fixed-time baseline -> DQN training -> test
evaluation -> CSV files -> figures -> (optional) PDF/PPTX report.

Examples
    python run_all.py --backend sumo --episodes 60          # real SUMO experiment
    python run_all.py --backend minisim --episodes 150      # built-in simulator (no SUMO)
    python run_all.py --quick                                # tiny smoke test
"""
import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from tsc.agents import torch_available
from tsc.config import DQNConfig, SimConfig, TrainConfig
from tsc.plots import make_plots
from tsc.runner import evaluate, load_best, make_agents, run_fixed_time, train

ROOT = Path(__file__).resolve().parent


def make_env(backend, cfg, gui=False):
    if backend == "sumo":
        from tsc.sumo_env import SumoEnv
        return SumoEnv(cfg, work_dir=ROOT / "sumo_files", gui=gui)
    from tsc.minisim_env import MiniSimEnv
    return MiniSimEnv(cfg)


def pick_backend(requested):
    if requested != "auto":
        return requested
    try:
        import traci  # noqa: F401
        from tsc.network import find_sumo_binary
        find_sumo_binary("sumo")
        return "sumo"
    except Exception:
        return "minisim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "sumo", "minisim"])
    ap.add_argument("--agent", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--test-episodes", type=int, default=None, help="number of unseen test seeds")
    ap.add_argument("--scales", type=float, nargs="+", default=None, help="demand multipliers for the test")
    ap.add_argument("--tune-baseline", action="store_true", help="also test fixed-time with 20 s and 40 s green")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    backend = pick_backend(args.backend)
    cfg, hp, tc = SimConfig(), DQNConfig(), TrainConfig()
    if args.quick:
        tc.episodes, tc.eval_every, tc.val_seeds, tc.test_seeds = 4, 2, (900,), (1000, 1001)
        cfg.sim_seconds = 600
        hp.learn_start = 100
    if args.episodes:
        tc.episodes = args.episodes
    if args.eval_every:
        tc.eval_every = args.eval_every
    if args.test_episodes:
        tc.test_seeds = tuple(range(1000, 1000 + args.test_episodes))
    if args.scales:
        tc.test_scales = tuple(args.scales)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agent_kind = args.agent if args.agent != "auto" else ("torch" if torch_available() else "numpy")
    print(f"backend={backend}  agent implementation={agent_kind}  episodes={tc.episodes}")

    env = make_env(backend, cfg)
    print("agents:", env.agents, "| actions per agent:", env.n_actions, "| lanes per agent:", env.n_lanes)
    t_start = time.time()

    # ---- fixed-time reference on validation seeds ------------------------------------
    fv = [run_fixed_time(env, s, 1.0, cfg.fixed_green) for s in tc.val_seeds]
    fixed_val = {k: float(np.mean([r[k] for r in fv])) for k in fv[0]}
    print("fixed-time validation:", {k: round(v, 2) for k, v in fixed_val.items()})

    # ---- training ------------------------------------------------------------------
    agents = make_agents(env, hp, agent_kind, seed=0)
    train_log, val_log = train(env, agents, hp, tc, out / "models")
    train_df, val_df = pd.DataFrame(train_log), pd.DataFrame(val_log)
    train_df.to_csv(out / "training_log.csv", index=False)
    val_df.to_csv(out / "validation_log.csv", index=False)
    load_best(agents, out / "models", env)
    best_info = json.loads((out / "models" / "best_info.json").read_text())
    print("best checkpoint:", best_info)

    # ---- final test on unseen seeds -------------------------------------------------
    rows = []
    green_options = [cfg.fixed_green] + ([20, 40] if args.tune_baseline else [])
    for scale in tc.test_scales:
        for g in green_options:
            for s in tc.test_seeds:
                m = run_fixed_time(env, s, scale, g)
                rows.append({"controller": f"Fixed-time ({g}s)", "demand_scale": scale, "seed": s, **m})
        for s, m in zip(tc.test_seeds, evaluate(env, agents, tc.test_seeds, scale)):
            rows.append({"controller": "Multi-agent DQN", "demand_scale": scale, "seed": s, **m})
    res = pd.DataFrame(rows)
    res["backend"] = backend
    res["agent_impl"] = agent_kind
    cols = ["controller", "demand_scale", "seed", "avg_wait_s", "avg_queue_per_lane", "avg_total_queue",
            "throughput", "departed", "cum_reward", "backend", "agent_impl"]
    res = res[cols]
    res.to_csv(out / "Experiment_Results.csv", index=False)
    summary = (res.groupby(["controller", "demand_scale"])
               [["avg_wait_s", "avg_queue_per_lane", "avg_total_queue", "throughput", "departed", "cum_reward"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(2).to_string())

    # ---- figures ---------------------------------------------------------------------
    main_scale = 1.0 if 1.0 in tc.test_scales else tc.test_scales[0]
    sel = res[res.demand_scale == main_scale]
    fx = sel[sel.controller == f"Fixed-time ({cfg.fixed_green}s)"]
    dq = sel[sel.controller == "Multi-agent DQN"]
    label = "SUMO" if backend == "sumo" else "built-in Python simulator (not SUMO)"
    make_plots(out, train_df, val_df, fixed_val, fx.to_dict("list"), dq.to_dict("list"), label)

    meta = {
        "backend": backend, "agent_impl": agent_kind, "python": platform.python_version(),
        "platform": platform.platform(), "main_scale": main_scale, "best_checkpoint": best_info,
        "runtime_s": round(time.time() - t_start, 1), "sim": asdict(cfg), "dqn": asdict(hp),
        "train": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(tc).items()},
        "fixed_val": fixed_val,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if hasattr(env, "close"):
        env.close()
    print(f"results written to {out}  ({meta['runtime_s']} s)")
    if not args.no_report:
        try:
            import make_report
            make_report.build(out)
        except ImportError:
            print("make_report.py not found - skipping report")


if __name__ == "__main__":
    sys.exit(main())
