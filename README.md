# Traffic-Signal Coordination with Multi-Agent DQN (Project 10)

Four signalised intersections (2 x 2 grid) are controlled by **four independent Double-DQN agents**
and compared with **fixed-time control**. Stack: Python 3.13, Eclipse SUMO 1.28, TraCI, PyTorch, NumPy, Pandas, Matplotlib.

## Results currently stored in `results/`
Backend that produced them: **built-in Python mesoscopic simulator (NOT SUMO)**; DQN implementation: **numpy**;
150 training episodes; mean over 10 unseen test seeds, demand scale 1.0.

| Metric | Fixed-time (30s) | Multi-agent DQN | Change |
|---|---|---|---|
| Average waiting time (s / vehicle) | 86.6 | 51.3 | -40.7% |
| Average queue length (halting veh / lane) | 2.23 | 1.28 | -42.5% |
| Vehicle throughput (veh completed / hour) | 2907 | 2946 | +1.3% |
| Cumulative reward (sum of 4 agents) | -5402 | -2570 | -52.4% |

> IMPORTANT: the numbers in this document were produced with the project's built-in Python mesoscopic simulator (tsc/minisim_env.py) because SUMO, TraCI and PyTorch were not available in the environment that built this report. The SUMO code path (tsc/sumo_env.py, tsc/network.py, tsc/agents/dqn_torch.py) is complete but has NOT been executed to produce these figures. Running 'python run_all.py --backend sumo' on a machine with SUMO 1.28 regenerates every table, figure, slide and this PDF from real SUMO runs.

## Run on Windows with VS Code
```powershell
cd C:\Projects\traffic_signal_marl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_setup.py                       # verifies SUMO, TraCI, 4 intersections, both controllers
python run_all.py --backend sumo --episodes 60 --test-episodes 10
```
If `sumo` is not found: `$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"`.
`--quick` runs a 2-minute smoke test; `--backend minisim` runs the built-in simulator without SUMO;
`--scales 1.0 0.7 1.3 --tune-baseline` adds demand levels and 20 s / 40 s fixed-time variants.

## Outputs (`results/`)
`Experiment_Results.csv` (one row per controller / demand / test seed), `summary.csv`, `training_log.csv`,
`validation_log.csv`, `Learning_Curves.png`, `performance_across_training.png`, `comparison_bars.png`,
`run_meta.json`, `models/`, `Traffic_Signal_Coordination_Complete_Submission.pdf`, `Five_Slide_Presentation.pptx`.
`python make_report.py` rebuilds the PDF, PPTX and this README from the stored results.

## Design in one paragraph
State per agent: density, queue and waiting time of every incoming lane + one-hot phase + time since switch + neighbours'
mean queue / wait. Action: choose the next green phase every 10 s (same = extend, different = 3 s yellow then switch).
Reward: `-(0.1 * halting + sum(wait)/300)`, shared with neighbours via `(1-a) r_i + a mean(r_nb)`, a = 0.3.
Metrics: average waiting time (vehicle-seconds standing / vehicles inserted), average queue (halting vehicles per lane),
throughput (completed trips per hour), cumulative reward (sum of local rewards, 4 agents).

## Code map
`tsc/sumo_env.py` SUMO+TraCI env | `tsc/minisim_env.py` fallback simulator | `tsc/network.py` network & routes |
`tsc/agents/` DQN (PyTorch + NumPy) | `tsc/runner.py` training / evaluation | `tsc/plots.py` figures | `run_all.py` pipeline.
