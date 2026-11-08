"""Built-in lightweight traffic simulator (fallback backend, NO SUMO needed).

IMPORTANT: this is a *simplified mesoscopic* model, not SUMO.  It exists so the whole
pipeline can be executed on machines where SUMO/TraCI is not available.  Results produced
with it must be reported as "built-in simulator", not as SUMO results.

Model summary (1 s time step):
  * 2x2 grid, 4 signalised junctions, each with 4 approaches; every approach has a
    through/right lane group (T) and a left lane group (L)  ->  8 lane groups / junction.
  * Vehicles travel along a link in a fixed free-flow time, then join the stop-line queue
    of their next movement.  A green movement discharges at its saturation flow rate
    after a 2 s start-up loss; yellow time discharges nothing.
  * Links have finite storage: a full downstream link blocks discharge (spill-back).
  * Same phases, decision interval, observation layout and reward as the SUMO backend.
"""
from collections import deque

import numpy as np

from .config import DIRS, JUNCTIONS, NEIGHBOURS, OPP, SimConfig, junction_rc
from .demand import border_points, od_table, rate_at

SIDES = "NESW"
CAP_MV = {"T": 30.0, "L": 15.0}        # nominal storage per lane group (vehicles)
SAT_RATE = {"T": 0.9, "L": 0.45}        # saturation discharge rate (veh/s)
LINK_CAP = 60                           # max vehicles on one link (storage)
STARTUP_LOST = 2                        # s of start-up loss after a phase turns green
CROSSING = 2                            # s to cross the junction
# Four signal phases: (approach sides that are green, movement group)
PHASES = [(("N", "S"), "T"), (("N", "S"), "L"), (("E", "W"), "T"), (("E", "W"), "L")]


class _Approach:
    """One approach (incoming link + stop line) of a junction."""
    __slots__ = ("q", "transit", "n_transit", "credit")

    def __init__(self):
        self.q = {"T": deque(), "L": deque()}
        self.transit = deque()                 # (arrival_time, vehicle)
        self.n_transit = {"T": 0, "L": 0}
        self.credit = {"T": 0.0, "L": 0.0}

    @property
    def storage(self):
        return len(self.transit) + len(self.q["T"]) + len(self.q["L"])


class MiniSimEnv:
    backend = "minisim"

    def __init__(self, cfg=None):
        self.cfg = cfg or SimConfig()
        self.agents = list(JUNCTIONS)
        self.n_actions = {a: len(PHASES) for a in self.agents}
        self.n_lanes = {a: 8 for a in self.agents}       # lane groups per agent
        self.entries = border_points()
        self.od = od_table()
        self.link_time = int(round(self.cfg.link_length / self.cfg.free_speed))
        self.total_lanes = sum(self.n_lanes.values())

    # ------------------------------------------------------------------ episode control
    def reset(self, seed=0, scale=1.0):
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.t = 0
        self.app = {(j, s): _Approach() for j in self.agents for s in SIDES}
        self.phase = {a: 0 for a in self.agents}
        self.pending = {a: 0 for a in self.agents}
        self.yellow_left = {a: 0 for a in self.agents}
        self.since = {a: 0 for a in self.agents}
        self.backlog = {e: deque() for e in self.entries}
        self.total_wait = 0.0
        self.departed = 0
        self.completed = 0
        self.q_samples = []
        return self._obs()

    def step(self, actions):
        """Apply one green-phase choice per agent and simulate `delta_time` seconds."""
        for a in self.agents:
            if actions[a] != self.phase[a] and self.yellow_left[a] == 0:
                self.pending[a] = actions[a]
                self.yellow_left[a] = self.cfg.yellow_time
        for _ in range(self.cfg.delta_time):
            self._tick()
        stats = {a: self._junction_stats(a) for a in self.agents}
        self.q_samples.append(sum(q for q, _ in stats.values()))
        local = {}
        for a, (q, w) in stats.items():
            r = -(self.cfg.reward_w_queue * q + self.cfg.reward_w_wait * w)
            local[a] = max(self.cfg.reward_floor, r)
        done = self.t >= self.cfg.sim_seconds
        return self._obs(), local, done, {}

    def metrics(self):
        mean_q = float(np.mean(self.q_samples)) if self.q_samples else 0.0
        return {
            "avg_wait_s": self.total_wait / max(1, self.departed),
            "avg_queue_per_lane": mean_q / self.total_lanes,
            "avg_total_queue": mean_q,
            "throughput": self.completed,
            "departed": self.departed,
        }

    def close(self):
        pass

    # ------------------------------------------------------------------ dynamics
    def _tick(self):
        t = self.t
        self._arrivals(t)
        for ap in self.app.values():
            while ap.transit and ap.transit[0][0] <= t:
                _, veh = ap.transit.popleft()
                mv = veh[0][veh[1]][3]
                ap.n_transit[mv] -= 1
                veh[2] = t
                ap.q[mv].append(veh)
        for a in self.agents:
            self._discharge(a, t)
        self.total_wait += sum(len(ap.q["T"]) + len(ap.q["L"]) for ap in self.app.values())
        self.t += 1

    def _arrivals(self, t):
        for e in self.entries:
            n = self.rng.poisson(rate_at(e[1], t, self.scale))
            for _ in range(n):
                options = self.od[e]
                route = options[int(self.rng.integers(len(options)))][1]
                self.backlog[e].append([route, 0, 0])
            bl = self.backlog[e]
            ap = self.app[(e[0], e[1])]
            while bl and ap.storage < LINK_CAP:
                veh = bl.popleft()
                mv = veh[0][0][3]
                ap.transit.append((t + self.link_time, veh))
                ap.n_transit[mv] += 1
                self.departed += 1

    def _discharge(self, a, t):
        if self.yellow_left[a] > 0:
            self.yellow_left[a] -= 1
            for s in SIDES:
                self.app[(a, s)].credit = {"T": 0.0, "L": 0.0}
            if self.yellow_left[a] == 0:
                self.phase[a] = self.pending[a]
                self.since[a] = 0
            return
        self.since[a] += 1
        sides, green_mv = PHASES[self.phase[a]]
        for s in SIDES:
            ap = self.app[(a, s)]
            for mv in ("T", "L"):
                if s in sides and mv == green_mv and self.since[a] > STARTUP_LOST:
                    ap.credit[mv] += SAT_RATE[mv]
                else:
                    ap.credit[mv] = 0.0
                    continue
                q = ap.q[mv]
                while q and ap.credit[mv] >= 1.0:
                    veh = q[0]
                    route, idx = veh[0], veh[1]
                    if idx == len(route) - 1:             # leaves the network
                        q.popleft()
                        self.completed += 1
                        ap.credit[mv] -= 1.0
                        continue
                    out = route[idx][2]
                    r, c = junction_rc(a)
                    nj = f"J{r + DIRS[out][0]}{c + DIRS[out][1]}"
                    tgt = self.app[(nj, OPP[out])]
                    if tgt.storage >= LINK_CAP:           # spill-back: blocked
                        break
                    q.popleft()
                    veh[1] = idx + 1
                    nmv = route[idx + 1][3]
                    tgt.transit.append((t + self.link_time + CROSSING, veh))
                    tgt.n_transit[nmv] += 1
                    ap.credit[mv] -= 1.0
                ap.credit[mv] = min(ap.credit[mv], 1.0)

    # ------------------------------------------------------------------ observations
    def _junction_stats(self, a):
        """(halting vehicles, summed waiting seconds) over the junction's approaches."""
        q_tot, w_tot = 0, 0.0
        for s in SIDES:
            for mv in ("T", "L"):
                q = self.app[(a, s)].q[mv]
                q_tot += len(q)
                w_tot += sum(self.t - v[2] for v in q)
        return q_tot, w_tot

    def _obs(self):
        """Per lane group: [density, queue, waiting] (all in 0..1), then one-hot phase and
        time since last phase change."""
        obs = {}
        for a in self.agents:
            f = []
            for s in SIDES:
                ap = self.app[(a, s)]
                for mv in ("T", "L"):
                    q = ap.q[mv]
                    cap = CAP_MV[mv]
                    f += [min(1.0, (len(q) + ap.n_transit[mv]) / cap),
                          min(1.0, len(q) / cap),
                          min(1.0, sum(self.t - v[2] for v in q) / 300.0)]
            onehot = [1.0 if i == self.phase[a] else 0.0 for i in range(len(PHASES))]
            f += onehot + [min(1.0, self.since[a] / 60.0)]
            obs[a] = np.asarray(f, dtype=np.float32)
        return obs
