"""Matplotlib figures built from the real experiment logs (no hand-entered numbers)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_DQN, C_FIX = "#1f6fb2", "#d1752b"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _smooth(y, k=10):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = np.concatenate([np.full(k - 1, y[0]), y])
    return np.convolve(pad, ker, mode="valid")


def _bar(ax, fixed, dqn, key, title, ylabel, fmt="{:.1f}"):
    f, d = fixed[key], dqn[key]
    means = [np.mean(f), np.mean(d)]
    stds = [np.std(f), np.std(d)]
    ax.bar([0, 1], means, yerr=stds, color=[C_FIX, C_DQN], width=0.55, capsize=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fixed-time", "Multi-agent DQN"], fontsize=8.5)
    for x, m in zip([0, 1], means):
        ax.text(x, m * 0.5, fmt.format(m), ha="center", color="white", fontsize=9, fontweight="bold")
    _style(ax, title, "", ylabel)
    ax.grid(axis="x", visible=False)


def make_plots(out_dir, train_df, val_df, fixed_val, fixed_test, dqn_test, backend_label):
    """train_df/val_df: DataFrames; fixed_val: dict of metric->mean on validation seeds;
    fixed_test/dqn_test: dict metric -> list of per-seed values on the test seeds."""
    out_dir = Path(out_dir)
    ep = train_df["episode"].values
    val_df = val_df[val_df["episode"] > 0]   # episode 0 = untrained net, off-scale; kept in the CSV

    # --- combined figure -------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2))
    a = ax[0, 0]
    a.plot(ep, train_df["cum_reward"], color=C_DQN, alpha=0.25, lw=1)
    a.plot(ep, _smooth(train_df["cum_reward"]), color=C_DQN, lw=2, label="DQN training (10-ep moving avg)")
    a.axhline(np.mean(fixed_test["cum_reward"]), color=C_FIX, ls="--", lw=1.8, label="Fixed-time (test mean)")
    a.legend(fontsize=7.5, frameon=False)
    _style(a, "(a) Training reward per episode", "Training episode", "Cumulative reward (sum of 4 agents)")

    for a, key, title, yl in [(ax[0, 1], "avg_wait_s", "(b) Avg waiting time during training", "Seconds per vehicle"),
                              (ax[0, 2], "avg_queue_per_lane", "(c) Avg queue length during training", "Halting vehicles per lane")]:
        a.plot(val_df["episode"], val_df[key], color=C_DQN, marker="o", ms=3.5, lw=1.8,
               label="Multi-agent DQN (greedy, validation)")
        a.axhline(fixed_val[key], color=C_FIX, ls="--", lw=1.8, label="Fixed-time (validation)")
        a.legend(fontsize=7.5, frameon=False)
        _style(a, title, "Training episode", yl)

    _bar(ax[1, 0], fixed_test, dqn_test, "avg_wait_s", "(d) Average waiting time (test)", "Seconds per vehicle")
    _bar(ax[1, 1], fixed_test, dqn_test, "avg_queue_per_lane", "(e) Average queue length (test)",
         "Halting vehicles per lane", "{:.2f}")
    _bar(ax[1, 2], fixed_test, dqn_test, "throughput", "(f) Vehicle throughput (test)",
         "Vehicles completed / hour", "{:.0f}")
    fig.suptitle(f"Fixed-time vs Multi-Agent DQN  |  backend: {backend_label}  |  mean +/- std over "
                 f"{len(dqn_test['avg_wait_s'])} unseen test seeds", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "Learning_Curves.png", dpi=150)
    plt.close(fig)

    # --- performance across training episodes (4 metrics) ----------------------------
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    items = [("cum_reward", "Cumulative reward"), ("avg_wait_s", "Avg waiting time (s)"),
             ("avg_queue_per_lane", "Avg queue (veh/lane)"), ("throughput", "Throughput (veh/h)")]
    for a, (k, t) in zip(ax, items):
        a.plot(val_df["episode"], val_df[k], color=C_DQN, marker="o", ms=3, lw=1.8, label="Multi-agent DQN")
        a.axhline(fixed_val[k], color=C_FIX, ls="--", lw=1.8, label="Fixed-time")
        _style(a, t, "Training episode", "")
    ax[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Performance across training episodes (greedy policy on validation seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "performance_across_training.png", dpi=150)
    plt.close(fig)

    # --- comparison bars (4 metrics incl. reward) ---------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    _bar(ax[0], fixed_test, dqn_test, "avg_wait_s", "Avg waiting time", "s / vehicle")
    _bar(ax[1], fixed_test, dqn_test, "avg_queue_per_lane", "Avg queue length", "veh / lane", "{:.2f}")
    _bar(ax[2], fixed_test, dqn_test, "throughput", "Throughput", "veh / hour", "{:.0f}")
    _bar(ax[3], fixed_test, dqn_test, "cum_reward", "Cumulative reward", "sum over 4 agents", "{:.0f}")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bars.png", dpi=150)
    plt.close(fig)
