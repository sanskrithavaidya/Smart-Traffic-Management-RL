"""Multi-agent SUMO environment driven through TraCI.

One agent per traffic light.  Every `delta_time` seconds each agent chooses which GREEN
phase should be active (choosing the current phase = extend it; choosing another phase =
yellow transition followed by that green).  The valid green phases are read from the
traffic-light program that `netconvert` generated, so the code is independent of the exact
signal layout.
"""
import os
import sys
import uuid
from pathlib import Path

import numpy as np

from .config import JUNCTIONS, SimConfig
from .network import build_network, find_sumo_binary, write_routes, write_sumocfg

try:                                     # normal case: `pip install traci`
    import traci
except ImportError:                      # fall back to SUMO's bundled python tools
    if "SUMO_HOME" in os.environ:
        sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import traci                         # noqa: E402  (raises a clear error if still missing)


class SumoEnv:
    backend = "sumo"

    def __init__(self, cfg=None, work_dir="sumo_files", gui=False):
        self.cfg = cfg or SimConfig()
        self.gui = gui
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.net = build_network(self.work_dir, self.cfg)
        write_sumocfg(self.work_dir / "grid2x2.sumocfg", self.net,
                      write_routes(self.work_dir / "routes_nominal.rou.xml", 0, 1.0))
        self.label = "sim_" + uuid.uuid4().hex[:8]
        self.conn = None
        self.agents = list(JUNCTIONS)
        self.n_actions, self.n_lanes = {}, {}
        self._introspect()

    # ------------------------------------------------------------------ SUMO control
    def _start(self, seed, scale):
        self.close()
        routes = write_routes(self.work_dir / f"routes_{self.label}.rou.xml", seed, scale)
        binary = find_sumo_binary("sumo-gui" if self.gui else "sumo")
        cmd = [binary, "-n", self.net, "-r", routes, "--seed", str(seed),
               "--no-step-log", "true", "--no-warnings", "true",
               "--duration-log.disable", "true", "--time-to-teleport", "300"]
        if self.gui:
            cmd += ["--start", "true", "--quit-on-end", "true"]
        traci.start(cmd, label=self.label)
        self.conn = traci.getConnection(self.label)

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def _introspect(self):
        """Start SUMO once to discover lanes and green phases of every traffic light."""
        self._start(0, 1.0)
        c = self.conn
        ids = set(c.trafficlight.getIDList())
        missing = [a for a in self.agents if a not in ids]
        if missing:
            raise RuntimeError(f"Traffic lights {missing} not found in the network; got {sorted(ids)}")
        self.lanes, self.lane_cap, self.greens = {}, {}, {}
        for a in self.agents:
            lanes = list(dict.fromkeys(c.trafficlight.getControlledLanes(a)))
            self.lanes[a] = lanes
            self.n_lanes[a] = len(lanes)
            for ln in lanes:
                self.lane_cap[ln] = max(1.0, c.lane.getLength(ln) / self.cfg.veh_space)
            logic = c.trafficlight.getAllProgramLogics(a)[0]
            greens = []
            for ph in logic.phases:
                s = ph.state
                if ("G" in s or "g" in s) and "y" not in s.lower() and s not in greens:
                    greens.append(s)
            if len(greens) < 2:
                raise RuntimeError(f"Traffic light {a} has fewer than 2 green phases: {logic.phases}")
            self.greens[a] = greens
            self.n_actions[a] = len(greens)
        self.edges = [e for e in c.edge.getIDList() if not e.startswith(":")]
        self.close()

    def reset(self, seed=0, scale=1.0):
        self._start(seed, scale)
        c = self.conn
        self.phase, self.since = {}, {}
        for a in self.agents:
            self.phase[a] = 0
            self.since[a] = 0
            c.trafficlight.setRedYellowGreenState(a, self.greens[a][0])
        self.total_wait, self.departed, self.completed = 0.0, 0, 0
        self.q_samples = []
        self.t = 0
        return self._obs()

    def _sim_second(self):
        c = self.conn
        c.simulationStep()
        self.t += 1
        self.departed += c.simulation.getDepartedNumber()
        self.completed += c.simulation.getArrivedNumber()
        # vehicle-seconds spent standing still (speed < 0.1 m/s) anywhere in the network
        self.total_wait += sum(c.edge.getLastStepHaltingNumber(e) for e in self.edges)

    def step(self, actions):
        c = self.conn
        cfg = self.cfg
        switching = {}
        for a in self.agents:
            if actions[a] != self.phase[a]:
                cur, nxt = self.greens[a][self.phase[a]], self.greens[a][actions[a]]
                yellow = "".join("y" if (x in "Gg" and y not in "Gg") else x
                                 for x, y in zip(cur, nxt))
                c.trafficlight.setRedYellowGreenState(a, yellow)
                switching[a] = actions[a]
        n_yellow = cfg.yellow_time if switching else 0
        for _ in range(n_yellow):
            self._sim_second()
        for a, new in switching.items():
            c.trafficlight.setRedYellowGreenState(a, self.greens[a][new])
            self.phase[a] = new
            self.since[a] = 0
        for _ in range(cfg.delta_time - n_yellow):
            self._sim_second()
        for a in self.agents:
            if a not in switching:
                self.since[a] += cfg.delta_time
            else:
                self.since[a] = cfg.delta_time - n_yellow
        # ---- statistics, reward
        local, total_q = {}, 0
        for a in self.agents:
            q = sum(c.lane.getLastStepHaltingNumber(ln) for ln in self.lanes[a])
            w = sum(c.lane.getWaitingTime(ln) for ln in self.lanes[a])
            total_q += q
            r = -(cfg.reward_w_queue * q + cfg.reward_w_wait * w)
            local[a] = max(cfg.reward_floor, r)
        self.q_samples.append(total_q)
        done = self.t >= cfg.sim_seconds
        obs = self._obs()
        if done:
            self._final = self.metrics()
            self.close()
        return obs, local, done, {}

    def _obs(self):
        """Per incoming lane: [density, queue, waiting] (0..1) then one-hot phase + time
        since last change.  Identical layout to the built-in simulator."""
        c = self.conn
        obs = {}
        for a in self.agents:
            f = []
            for ln in self.lanes[a]:
                cap = self.lane_cap[ln]
                f += [min(1.0, c.lane.getLastStepVehicleNumber(ln) / cap),
                      min(1.0, c.lane.getLastStepHaltingNumber(ln) / cap),
                      min(1.0, c.lane.getWaitingTime(ln) / 300.0)]
            k = self.n_actions[a]
            f += [1.0 if i == self.phase[a] else 0.0 for i in range(k)]
            f += [min(1.0, self.since[a] / 60.0)]
            obs[a] = np.asarray(f, dtype=np.float32)
        return obs

    def metrics(self):
        if self.conn is None and hasattr(self, "_final"):
            return self._final
        mean_q = float(np.mean(self.q_samples)) if self.q_samples else 0.0
        total_lanes = sum(self.n_lanes.values())
        return {
            "avg_wait_s": self.total_wait / max(1, self.departed),
            "avg_queue_per_lane": mean_q / total_lanes,
            "avg_total_queue": mean_q,
            "throughput": self.completed,
            "departed": self.departed,
        }
