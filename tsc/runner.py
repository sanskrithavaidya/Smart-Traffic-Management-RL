"""Episode runners: fixed-time baseline, multi-agent DQN training and greedy evaluation.

Coordination between the independent DQN agents happens in two ways (see README):
  1. every agent's observation is extended with the mean queue / waiting level of its
     neighbouring intersections ("neighbour summary"), and
  2. every agent is trained on a *shared* reward that mixes its own reward with the mean
     reward of its neighbours:  r_i = (1 - alpha) * r_i + alpha * mean_j(r_j).
"""
import json
import time
from pathlib import Path

import numpy as np

from .agents import make_agent
from .config import NEIGHBOURS


# ------------------------------------------------------------------ coordination helpers
def augment(env, obs):
    """Append the neighbours' mean (queue, waiting) level to every agent's observation."""
    summ = {}
    for a, o in obs.items():
        lane = o[:3 * env.n_lanes[a]].reshape(-1, 3)
        summ[a] = lane[:, 1:3].mean(axis=0)                  # mean queue, mean waiting
    out = {}
    for a, o in obs.items():
        nb = np.mean([summ[b] for b in NEIGHBOURS[a]], axis=0)
        out[a] = np.concatenate([o, nb]).astype(np.float32)
    return out


def shared_rewards(local, alpha):
    """Mix own reward with the neighbours' mean reward (cooperative reward shaping)."""
    return {a: (1 - alpha) * r + alpha * float(np.mean([local[b] for b in NEIGHBOURS[a]]))
            for a, r in local.items()}


# ------------------------------------------------------------------ episodes
def run_fixed_time(env, seed, scale=1.0, green_s=30):
    """Fixed-time control: cycle through all green phases, `green_s` seconds each."""
    hold = max(1, int(round(green_s / env.cfg.delta_time)))
    env.reset(seed, scale)
    step, cum = 0, 0.0
    done = False
    while not done:
        acts = {a: (step // hold) % env.n_actions[a] for a in env.agents}
        _, local, done, _ = env.step(acts)
        cum += sum(local.values())
        step += 1
    m = dict(env.metrics())
    m["cum_reward"] = cum
    return m


def run_dqn_eval(env, agents, seed, scale=1.0):
    """Greedy (epsilon = 0) execution of the trained agents."""
    obs = augment(env, env.reset(seed, scale))
    cum, done = 0.0, False
    while not done:
        acts = {a: agents[a].act(obs[a], 0.0) for a in env.agents}
        nobs, local, done, _ = env.step(acts)
        obs = augment(env, nobs)
        cum += sum(local.values())
    m = dict(env.metrics())
    m["cum_reward"] = cum
    return m


def make_agents(env, hp, kind="auto", seed=0):
    probe = augment(env, env.reset(0, 1.0))
    agents = {a: make_agent(len(probe[a]), env.n_actions[a], hp, seed + i, kind)
              for i, a in enumerate(env.agents)}
    if hasattr(env, "close"):
        env.close()
    return agents


def evaluate(env, agents, seeds, scale=1.0):
    return [run_dqn_eval(env, agents, s, scale) for s in seeds]


# ------------------------------------------------------------------ training
def train(env, agents, hp, tc, out_dir, log=print):
    """Train all agents simultaneously.  Returns (train_log, validation_log)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    train_log, val_log = [], []
    best_val = float("inf")
    decay_eps = max(1, int(tc.episodes * hp.eps_decay_frac))
    t0 = time.time()

    def validate(ep):
        res = evaluate(env, agents, tc.val_seeds, 1.0)
        row = {"episode": ep}
        for k in res[0]:
            row[k] = float(np.mean([r[k] for r in res]))
        val_log.append(row)
        return row

    row = validate(0)                                        # untrained policy reference
    for ep in range(1, tc.episodes + 1):
        eps = max(hp.eps_end, hp.eps_start - (hp.eps_start - hp.eps_end) * ep / decay_eps)
        scale = float(rng.uniform(*tc.train_scale_range))
        obs = augment(env, env.reset(tc.train_seed0 + ep, scale))
        done, cum_local, cum_shared, losses = False, 0.0, 0.0, []
        while not done:
            acts = {a: agents[a].act(obs[a], eps) for a in env.agents}
            nobs, local, done, _ = env.step(acts)
            nobs = augment(env, nobs)
            shared = shared_rewards(local, hp.coop_alpha)
            for a in env.agents:
                # episode end is a time limit, not a terminal state -> bootstrap (done=0)
                agents[a].store(obs[a], acts[a], shared[a], nobs[a], 0.0)
                loss = agents[a].learn()
                if loss is not None:
                    losses.append(loss)
            obs = nobs
            cum_local += sum(local.values())
            cum_shared += sum(shared.values())
        m = dict(env.metrics())
        m.update(episode=ep, epsilon=eps, demand_scale=scale, cum_reward=cum_local,
                 cum_shared_reward=cum_shared,
                 loss=float(np.mean(losses)) if losses else float("nan"))
        train_log.append(m)
        msg = (f"ep {ep:4d}/{tc.episodes} eps={eps:.2f} scale={scale:.2f} "
               f"reward={cum_local:9.1f} wait={m['avg_wait_s']:6.1f}s "
               f"queue={m['avg_queue_per_lane']:.2f} thr={m['throughput']} "
               f"[{time.time() - t0:.0f}s]")
        if ep % tc.eval_every == 0 or ep == tc.episodes:
            v = validate(ep)
            msg += f" | VAL wait={v['avg_wait_s']:.1f}s reward={v['cum_reward']:.1f}"
            if v["avg_wait_s"] < best_val:                   # keep best checkpoint (validation seeds)
                best_val = v["avg_wait_s"]
                for a in env.agents:
                    agents[a].save(out_dir / f"best_{a}")
                (out_dir / "best_info.json").write_text(json.dumps({"episode": ep, "val_avg_wait_s": best_val}))
        log(msg)
    for a in env.agents:
        agents[a].save(out_dir / f"final_{a}")
    return train_log, val_log


def load_best(agents, out_dir, env):
    for a in env.agents:
        agents[a].load(Path(out_dir) / f"best_{a}")
