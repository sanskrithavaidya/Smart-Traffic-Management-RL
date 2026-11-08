"""Generate the SUMO road network (2x2 signalised grid) and the route/demand files.

Files are produced with plain XML + `netconvert`, so no SUMO tool-specific templates are
required.  Naming convention:
    junction nodes   J00 J01 J10 J11          (traffic lights, same ids)
    border nodes     B_<junction>_<side>
    entry edges      in_<junction>_<side>      (border  -> junction)
    exit edges       out_<junction>_<side>     (junction -> border)
    inner edges      e_<from>_<to>
"""
import os
import shutil
import subprocess
from pathlib import Path

from .config import DIRS, JUNCTIONS, SimConfig, junction_rc
from .demand import border_points, entry_edge, exit_edge, od_table, rate_at
from .config import DEMAND_PROFILE


def find_sumo_binary(name="sumo"):
    """Locate a SUMO executable (works on Windows with SUMO_HOME or PATH)."""
    exe = name + (".exe" if os.name == "nt" else "")
    home = os.environ.get("SUMO_HOME")
    if home:
        cand = Path(home) / "bin" / exe
        if cand.exists():
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Cannot find '{name}'. Set SUMO_HOME (e.g. C:\\Program Files (x86)\\Eclipse\\Sumo) "
        "or add the SUMO bin folder to PATH.")


def _xy(j, cfg):
    r, c = junction_rc(j)
    return c * cfg.link_length, -r * cfg.link_length


def build_network(out_dir, cfg=None):
    """Write nodes/edges XML and run netconvert.  Returns the .net.xml path."""
    cfg = cfg or SimConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes, edges = [], []
    for j in JUNCTIONS:
        x, y = _xy(j, cfg)
        nodes.append(f'  <node id="{j}" x="{x}" y="{y}" type="traffic_light"/>')
    e_attr = f'numLanes="{cfg.lanes_per_edge}" speed="{cfg.free_speed}"'
    for (j, side) in border_points():
        x, y = _xy(j, cfg)
        dr, dc = DIRS[side]
        bx, by = x + dc * cfg.link_length, y - dr * cfg.link_length
        b = f"B_{j}_{side}"
        nodes.append(f'  <node id="{b}" x="{bx}" y="{by}" type="priority"/>')
        edges.append(f'  <edge id="in_{j}_{side}" from="{b}" to="{j}" {e_attr}/>')
        edges.append(f'  <edge id="out_{j}_{side}" from="{j}" to="{b}" {e_attr}/>')
    for a in JUNCTIONS:
        ra, ca = junction_rc(a)
        for b in JUNCTIONS:
            rb, cb = junction_rc(b)
            if abs(ra - rb) + abs(ca - cb) == 1:
                edges.append(f'  <edge id="e_{a}_{b}" from="{a}" to="{b}" {e_attr}/>')
    nod = out_dir / "nodes.nod.xml"
    edg = out_dir / "edges.edg.xml"
    nod.write_text('<nodes>\n' + "\n".join(nodes) + '\n</nodes>\n')
    edg.write_text('<edges>\n' + "\n".join(edges) + '\n</edges>\n')
    net = out_dir / "grid2x2.net.xml"
    cmd = [find_sumo_binary("netconvert"), "--node-files", str(nod), "--edge-files", str(edg),
           "--output-file", str(net), "--no-turnarounds", "true", "--no-warnings", "true"]
    subprocess.run(cmd, check=True)
    return str(net)


def write_routes(path, seed=0, scale=1.0):
    """Write a demand file: Poisson flows for every valid origin-destination pair with a
    three-segment (off-peak / peak / off-peak) demand profile.  `seed` only changes the
    SUMO random seed (given on the command line), the file itself is deterministic."""
    table = od_table()
    lines = ['<routes>',
             '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" '
             'maxSpeed="13.89"/>']
    for seg_i, (b, e, f) in enumerate(DEMAND_PROFILE):
        k = 0
        for entry, options in table.items():
            side = entry[1]
            per_od = rate_at(side, b, scale) * 3600.0 / len(options)   # veh/h per OD pair
            if per_od <= 0:
                continue
            for exit_, _route in options:
                lines.append(
                    f'  <flow id="f{seg_i}_{k}" type="car" begin="{b}" end="{e}" '
                    f'period="exp({per_od / 3600.0:.8f})" from="{entry_edge(entry)}" '
                    f'to="{exit_edge(exit_)}" departLane="best" departSpeed="max"/>')
                k += 1
    lines.append('</routes>')
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def write_sumocfg(path, net, routes):
    """Optional .sumocfg so the scenario can be opened in sumo-gui for visual inspection."""
    Path(path).write_text(
        '<configuration>\n  <input>\n'
        f'    <net-file value="{Path(net).name}"/>\n'
        f'    <route-files value="{Path(routes).name}"/>\n'
        '  </input>\n  <time>\n    <begin value="0"/>\n    <end value="3600"/>\n  </time>\n'
        '</configuration>\n')
    return str(path)
