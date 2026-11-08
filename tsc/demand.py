"""Origin-destination demand shared by the SUMO and the built-in simulator backends.

Vehicles enter the 2x2 grid from 8 border entry points and leave through one of the
other border exit points.  Routes are "horizontal first, then vertical".
"""
from .config import (DEMAND_PROFILE, DIRS, ENTRY_RATE_VPH, JUNCTIONS, LEFT, OPP,
                     junction_rc)


def border_points():
    """All (junction, side) pairs whose outer side has no neighbouring junction."""
    pts = []
    for j in JUNCTIONS:
        r, c = junction_rc(j)
        for side, (dr, dc) in DIRS.items():
            if f"J{r + dr}{c + dc}" not in JUNCTIONS:
                pts.append((j, side))
    return pts


def movement_type(h_in, h_out):
    """'T' through, 'L' left, 'R' right (right turns share the through signal group)."""
    if h_in == h_out:
        return "T"
    return "L" if LEFT[h_in] == h_out else "R"


def compute_route(entry, exit_):
    """Route as a list of (junction, heading_in, heading_out, movement) or None if invalid
    (a route that would need a U-turn)."""
    (j0, side0), (j1, side1) = entry, exit_
    r, c = junction_rc(j0)
    tr, tc = junction_rc(j1)
    heading = OPP[side0]
    moves = []
    while (r, c) != (tr, tc):
        if c != tc:
            out = "E" if tc > c else "W"
        else:
            out = "S" if tr > r else "N"
        moves.append([f"J{r}{c}", heading, out])
        r, c = r + DIRS[out][0], c + DIRS[out][1]
        heading = out
    moves.append([f"J{r}{c}", heading, side1])
    for _, h_in, h_out in moves:
        if h_out == OPP[h_in]:
            return None
    return [(j, hi, ho, "L" if movement_type(hi, ho) == "L" else "T") for j, hi, ho in moves]


def od_table():
    """{entry: [(exit, route), ...]} containing every valid origin-destination pair."""
    pts = border_points()
    table = {}
    for e in pts:
        lst = []
        for x in pts:
            route = compute_route(e, x)
            if route is not None:
                lst.append((x, route))
        table[e] = lst
    return table


def rate_at(side, t, scale=1.0):
    """Entry rate (veh/s) at time t for an entry on `side`."""
    for b, e, f in DEMAND_PROFILE:
        if b <= t < e:
            return ENTRY_RATE_VPH[side] * f * scale / 3600.0
    return 0.0


def entry_edge(entry):
    return f"in_{entry[0]}_{entry[1]}"


def exit_edge(exit_):
    return f"out_{exit_[0]}_{exit_[1]}"
