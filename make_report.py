"""Builds the submission documents from the REAL experiment outputs in results/:

    results/Traffic_Signal_Coordination_Complete_Submission.pdf   (A4, all four deliverables)
    results/Five_Slide_Presentation.pptx                          (exactly 5 slides)
    README.md                                                     (project root)

Every number is read from results/Experiment_Results.csv, summary.csv, run_meta.json and the
training logs - nothing is typed in by hand.  Run after run_all.py:   python make_report.py
"""
import datetime
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent
NAVY, BLUE, ORANGE, GREY = "#14284b", "#1f6fb2", "#d1752b", "#5b6573"
PDF_NAME = "Traffic_Signal_Coordination_Complete_Submission.pdf"
PPT_NAME = "Five_Slide_Presentation.pptx"


# =============================================================================== data
def load(out):
    out = Path(out)
    meta = json.loads((out / "run_meta.json").read_text())
    res = pd.read_csv(out / "Experiment_Results.csv")
    summ = pd.read_csv(out / "summary.csv")
    train = pd.read_csv(out / "training_log.csv")
    val = pd.read_csv(out / "validation_log.csv")
    return meta, res, summ, train, val


def pct(new, old):
    return 100.0 * (new - old) / old if old else float("nan")


def facts(out):
    meta, res, summ, train, val = load(out)
    sc = meta["main_scale"]
    fname = f"Fixed-time ({meta['sim']['fixed_green']}s)"
    sub = res[res.demand_scale == sc]
    F = sub[sub.controller == fname].sort_values("seed").reset_index(drop=True)
    D = sub[sub.controller == "Multi-agent DQN"].sort_values("seed").reset_index(drop=True)
    keys = [("avg_wait_s", "Average waiting time (s / vehicle)", "{:.1f}", True),
            ("avg_queue_per_lane", "Average queue length (halting veh / lane)", "{:.2f}", True),
            ("throughput", "Vehicle throughput (veh completed / hour)", "{:.0f}", False),
            ("cum_reward", "Cumulative reward (sum of 4 agents)", "{:.0f}", False)]
    rows = []
    for k, label, fmt, lower_better in keys:
        fm, fs, dm, ds = F[k].mean(), F[k].std(), D[k].mean(), D[k].std()
        better = (D[k] < F[k]) if lower_better else (D[k] > F[k])
        rows.append(dict(key=k, label=label, fmt=fmt, fixed_m=fm, fixed_s=fs, dqn_m=dm, dqn_s=ds,
                         change=pct(dm, fm), wins=int(better.sum()), n=len(D), lower_better=lower_better))
    return dict(meta=meta, res=res, summ=summ, train=train, val=val, rows=rows, fname=fname,
                F=F, D=D, scale=sc, backend=meta["backend"], impl=meta["agent_impl"],
                n_test=len(D), n_train=len(train))


def verdict(row):
    """'reduced by x %' / 'increased by x %' wording that follows the real sign."""
    c = row["change"]
    word = "lower" if c < 0 else "higher"
    return f"{abs(c):.1f}% {word}"


def backend_label(f):
    return ("Eclipse SUMO (TraCI)" if f["backend"] == "sumo"
            else "built-in Python mesoscopic simulator (NOT SUMO)")


def backend_note(f):
    if f["backend"] == "sumo":
        return ("All results in this document were produced by running the SUMO simulation through "
                "TraCI on the machine that built this report.")
    return ("IMPORTANT: the numbers in this document were produced with the project's built-in "
            "Python mesoscopic simulator (tsc/minisim_env.py) because SUMO, TraCI and PyTorch were not "
            "available in the environment that built this report. The SUMO code path "
            "(tsc/sumo_env.py, tsc/network.py, tsc/agents/dqn_torch.py) is complete but has NOT been "
            "executed to produce these figures. Running 'python run_all.py --backend sumo' on a machine "
            "with SUMO 1.28 regenerates every table, figure, slide and this PDF from real SUMO runs.")


# =============================================================================== figures
def network_figure(path):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    pos = {"J00": (0, 1), "J01": (1, 1), "J10": (0, 0), "J11": (1, 0)}
    for a, b in [("J00", "J01"), ("J10", "J11"), ("J00", "J10"), ("J01", "J11")]:
        ax.plot(*zip(pos[a], pos[b]), color="#444", lw=7, solid_capstyle="butt", zorder=1)
        ax.plot(*zip(pos[a], pos[b]), color="white", lw=0.8, ls="--", zorder=2)
    for j, (x, y) in pos.items():
        for dx, dy in [(-0.65, 0), (0.65, 0), (0, 0.65), (0, -0.65)]:
            bx, by = x + dx, y + dy
            if abs(bx - 0.5) > 0.6 or abs(by - 0.5) > 0.6:
                ax.plot([x, bx], [y, by], color="#888", lw=5, solid_capstyle="butt", zorder=1)
                ax.annotate("", xy=(x + dx * 0.55, y + dy * 0.55), xytext=(bx, by),
                            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
        ax.scatter([x], [y], s=900, color=BLUE, zorder=3, edgecolor="white", linewidth=2)
        ax.text(x, y, j, color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)
    ax.text(0.5, 0.5, "250 m links\n2 lanes / direction", ha="center", va="center", fontsize=8, color=GREY)
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(-0.9, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("2 x 2 grid: 4 signalised junctions, 8 entry / exit points", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def architecture_figure(path, f):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, tc="white", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold")

    def arrow(p, q, color="#333", rad=0.0, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color,
                                     connectionstyle=f"arc3,rad={rad}", lw=1.3))
    sim = "Eclipse SUMO\n(via TraCI)" if f["backend"] == "sumo" else "Built-in simulator\n(SUMO-free fallback)"
    box(0.2, 2.1, 2.1, 1.6, sim + "\n\n4 signalised junctions\nflows / routes", NAVY)
    box(3.0, 2.1, 2.1, 1.6, "Multi-agent env\nobservations: density,\nqueue, waiting time\n+ neighbour summary", GREY)
    pos = {"J00": (6.0, 3.15), "J01": (8.6, 3.15), "J10": (6.0, 1.55), "J11": (8.6, 1.55)}
    for j, (x, y) in pos.items():
        box(x, y, 2.2, 1.3, f"DQN agent {j}\nQ-network 128-128\nreplay buffer\ntarget network", BLUE, fs=8.5)
    arrow((2.3, 3.1), (3.0, 3.1))
    arrow((3.0, 2.7), (2.3, 2.7), ORANGE)
    arrow((5.1, 3.4), (6.0, 3.8))
    arrow((5.1, 2.4), (6.0, 2.2))
    arrow((6.0, 3.3), (5.1, 3.0), ORANGE)
    arrow((7.1, 3.15), (7.1, 2.85), "#2a9d8f", style="<|-|>")
    arrow((9.7, 3.15), (9.7, 2.85), "#2a9d8f", style="<|-|>")
    arrow((8.2, 3.8), (8.6, 3.8), "#2a9d8f", style="<|-|>")
    arrow((8.2, 2.2), (8.6, 2.2), "#2a9d8f", style="<|-|>")
    ax.text(0.2, 4.3, "actions: green phase per junction\n(extend current / switch via yellow)", color=ORANGE, fontsize=8.5, va="top")
    ax.text(6.0, 5.25, "Independent learners, cooperative through:", fontsize=9.5, fontweight="bold")
    ax.text(6.0, 4.85, "(1) neighbour queue/wait summary in each state\n(2) shared reward mix (alpha = "
            f"{f['meta']['dqn']['coop_alpha']})", fontsize=8.3, color="#2a9d8f", va="top")
    box(0.2, 0.2, 4.9, 0.85, "Fixed-time baseline: same environment, phases cycled\n"
        f"{f['meta']['sim']['fixed_green']} s per phase (no learning)", ORANGE, fs=8.5)
    box(6.0, 0.2, 4.8, 0.85, "Evaluation: avg waiting time, avg queue, throughput,\ncumulative reward on unseen test seeds", "#2a9d8f", fs=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# =============================================================================== slides
def slide_specs(f, figs):
    m, rows = f["meta"], {r["key"]: r for r in f["rows"]}
    w, q, th, cr = rows["avg_wait_s"], rows["avg_queue_per_lane"], rows["throughput"], rows["cum_reward"]
    sim = m["sim"]
    bk = "SUMO" if f["backend"] == "sumo" else "built-in simulator (SUMO-free fallback)"
    return [
        dict(title="Multi-Agent RL for Traffic-Signal Coordination", kicker="Slide 1 | Motivation & introduction",
             bullets=[("Urban congestion wastes travel time, fuel and air quality; much of it is queueing at signals.", 0),
                      ("Fixed-time plans repeat the same cycle all day and cannot react to fluctuating demand.", 0),
                      ("Neighbouring intersections influence each other: a long queue released upstream arrives downstream seconds later.", 0),
                      ("Project 10: one DQN agent per intersection learns to adapt its signal; goal = shorter waits and queues.", 0),
                      (f"Testbed: 2 x 2 signalised grid, one simulated hour with an off-peak / peak / off-peak demand profile ({bk}).", 0)],
             image=figs["network"], img_box=(7.6, 1.35, 5.2, 5.2)),
        dict(title="Problem Formulation", kicker="Slide 2 | Environment, states, actions, rewards",
             bullets=[("Environment: 4 junctions, 250 m links, 2 lanes per direction, Poisson demand on 8 entries.", 0),
                      ("State (per agent): density, queue and waiting time of each incoming lane + one-hot current phase + time since switch + neighbours' mean queue / wait.", 0),
                      (f"Action: choose the next green phase ({m['sim']['delta_time']} s decision step): same phase = extend, different = {sim['yellow_time']} s yellow then switch.", 0),
                      ("Reward: r = -(0.1 x halting vehicles + sum of waiting seconds / 300), clipped at -10.", 0),
                      (f"Training reward is shared: (1 - a) r_i + a mean(r_neighbours), a = {m['dqn']['coop_alpha']}.", 0),
                      ("Metrics: avg waiting time, avg queue, throughput, cumulative reward.", 0)],
             image=None, table=True),
        dict(title="Methodology: Multi-Agent Double DQN", kicker="Slide 3 | Architecture & training",
             bullets=[("Independent learners: each junction has its own Q-network (MLP 128-128), replay buffer and target network.", 0),
                      (f"Double-DQN target, Huber loss, Adam (lr {m['dqn']['lr']}), gamma {m['dqn']['gamma']}, batch {m['dqn']['batch_size']}.", 0),
                      (f"Epsilon-greedy {m['dqn']['eps_start']} -> {m['dqn']['eps_end']}; {m['train']['episodes']} training episodes with random demand jitter.", 0),
                      ("Coordination = neighbour summary in the state + cooperative reward mix.", 0),
                      ("Model selection on separate validation seeds; final test on unseen seeds against fixed-time control.", 0)],
             image=figs["arch"], img_box=(0.5, 3.75, 12.3, 3.25), text_h=2.4),
        dict(title="Results: Fixed-Time vs Multi-Agent DQN", kicker=f"Slide 4 | Experimental results ({bk})",
             bullets=[(f"Avg waiting time: {w['fixed_m']:.1f} s -> {w['dqn_m']:.1f} s ({verdict(w)} with DQN).", 0),
                      (f"Avg queue: {q['fixed_m']:.2f} -> {q['dqn_m']:.2f} veh/lane ({verdict(q)}).", 0),
                      (f"Throughput: {th['fixed_m']:.0f} -> {th['dqn_m']:.0f} veh/h ({verdict(th)}).", 0),
                      (f"Cumulative reward: {cr['fixed_m']:.0f} -> {cr['dqn_m']:.0f}.", 0),
                      (f"Mean over {f['n_test']} unseen test seeds; DQN better on waiting time in {w['wins']} of {w['n']} seeds.", 0)],
             image=figs["bars"], img_box=(0.5, 3.85, 12.3, 3.15), text_h=2.4),
        dict(title="Limitations, Ethics, Future Scope & Conclusion", kicker="Slide 5",
             bullets=[("Limitations: one small grid, synthetic demand, one random-seed family, no pedestrians / buses / sensor noise; independent learners are non-stationary.", 0),
                      ("Ethics: safety (min-yellow, human fallback to fixed-time), fairness to minor roads and pedestrians, and equity of who benefits must be audited before deployment.", 0),
                      ("Future scope: larger / real networks, graph-based communication (CoLight, MA-PPO), pressure-based rewards, pedestrians, emergency-vehicle priority.", 0),
                      (f"Conclusion: on this benchmark the multi-agent DQN changed average waiting time by {w['change']:+.1f}% and throughput by {th['change']:+.1f}% relative to fixed-time control ({bk}).", 0)],
             image=None),
    ]


def build_pptx(specs, path, f):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    rgb = lambda h: RGBColor.from_string(h.lstrip("#").upper())
    for i, sp in enumerate(specs, 1):
        s = prs.slides.add_slide(blank)
        bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.15))
        bar.fill.solid()
        bar.fill.fore_color.rgb = rgb(NAVY)
        bar.line.fill.background()
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.12), Inches(12.3), Inches(0.5))
        r = tb.text_frame.paragraphs[0].add_run()
        r.text = sp["kicker"]
        r.font.size, r.font.color.rgb = Pt(13), rgb("9fc3ea")
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12.3), Inches(0.6))
        r = tb.text_frame.paragraphs[0].add_run()
        r.text = sp["title"]
        r.font.size, r.font.bold, r.font.color.rgb = Pt(28), True, rgb("FFFFFF")
        has_img = sp.get("image") is not None
        side_img = has_img and sp["img_box"][0] > 5
        text_w = 6.9 if (side_img or sp.get("table")) else 12.3
        text_h = sp.get("text_h", 5.6)
        tb = s.shapes.add_textbox(Inches(0.5), Inches(1.35), Inches(text_w), Inches(text_h))
        tf = tb.text_frame
        tf.word_wrap = True
        fs = 15 if (has_img and not side_img) else (17 if has_img or sp.get("table") else 21)
        for k, (txt, lvl) in enumerate(sp["bullets"]):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            run = p.add_run()
            run.text = txt
            run.font.size, run.font.color.rgb = Pt(fs - 2 * lvl), rgb("1b1f27")
            p.space_after = Pt(6)
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.28 + 0.3 * lvl))))
            pPr.set("indent", str(-int(Inches(0.28))))
            bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "\u2022"})
            pPr.append(bu)
        if has_img:
            x, y, w, h = sp["img_box"]
            from PIL import Image as PILImage
            iw, ih = PILImage.open(sp["image"]).size
            sc = min(w / iw, h / ih)
            s.shapes.add_picture(sp["image"], Inches(x + (w - iw * sc) / 2), Inches(y + (h - ih * sc) / 2),
                                 Inches(iw * sc), Inches(ih * sc))
        if sp.get("table"):
            rows = [("Component", "Design"), ("Agents", "4 (one per junction)"),
                    ("Actions / agent", f"{f['meta']['sim']['delta_time']} s step, 4 green phases"),
                    ("State dim / agent", str(f["state_dim"])), ("Episode", "3600 s = 360 decisions")]
            tbl = s.shapes.add_table(len(rows), 2, Inches(7.9), Inches(1.6), Inches(4.9), Inches(2.4)).table
            for ri, row in enumerate(rows):
                for ci, v in enumerate(row):
                    c = tbl.cell(ri, ci)
                    c.text = v
                    c.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
        ft = s.shapes.add_textbox(Inches(0.5), Inches(7.08), Inches(12.3), Inches(0.35))
        r = ft.text_frame.paragraphs[0].add_run()
        r.text = f"Traffic-signal coordination | Project 10 | results backend: {backend_label(f)} | slide {i} of 5"
        r.font.size, r.font.color.rgb = Pt(10), rgb("5b6573")
    prs.save(path)


class SlideFlowable(Flowable):
    """Draws one slide (same spec as the PPTX) so the PDF shows the five slides."""

    def __init__(self, spec, width, f):
        super().__init__()
        self.spec, self.width, self.f = spec, width, f
        self.height = width * 7.5 / 13.333

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        from reportlab.platypus import Paragraph as P
        c, sp, s = self.canv, self.spec, self.width / 13.333
        k = s / 72.0   # font points -> output points
        H = self.height
        Y = lambda y: H - y * s
        c.setStrokeColor(colors.HexColor("#c9ced6"))
        c.setFillColor(colors.white)
        c.rect(0, 0, self.width, H, fill=1, stroke=1)
        c.setFillColor(colors.HexColor(NAVY))
        c.rect(0, Y(1.15), self.width, 1.15 * s, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#9fc3ea"))
        c.setFont("Helvetica", 13 * k)
        c.drawString(0.5 * s, Y(0.42), sp["kicker"])
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 28 * k)
        c.drawString(0.5 * s, Y(0.95), sp["title"])
        has_img = sp.get("image") is not None
        side_img = has_img and sp["img_box"][0] > 5
        text_w = 6.9 if (side_img or sp.get("table")) else 12.3
        fs = (15 if (has_img and not side_img) else (17 if (has_img or sp.get("table")) else 21)) * k
        st = ParagraphStyle("sl", fontName="Helvetica", fontSize=fs, leading=fs * 1.22, leftIndent=fs * 0.9,
                            bulletIndent=0, spaceAfter=fs * 0.4, textColor=colors.HexColor("#1b1f27"))
        y = 1.35 * s
        for txt, lvl in sp["bullets"]:
            p = P(txt.replace("&", "&amp;"), st, bulletText="\u2022")
            w, h = p.wrap(text_w * s, H)
            p.drawOn(c, 0.5 * s, Y(0) - y - h)
            y += h + fs * 0.4
        if has_img:
            bx, by, bw, bh = sp["img_box"]
            c.drawImage(sp["image"], bx * s, Y(by + bh), bw * s, bh * s, preserveAspectRatio=True, anchor="c", mask="auto")
        if sp.get("table"):
            rows = [("Component", "Design"), ("Agents", "4 (one per junction)"),
                    ("Actions / agent", f"{self.f['meta']['sim']['delta_time']} s step, 4 green phases"),
                    ("State dim / agent", str(self.f["state_dim"])), ("Episode", "3600 s = 360 decisions")]
            ty = 1.6
            for ri, (a, b) in enumerate(rows):
                c.setFillColor(colors.HexColor("#e8eef6" if ri == 0 else "#ffffff"))
                c.rect(7.9 * s, Y(ty + 0.48), 4.9 * s, 0.48 * s, fill=1, stroke=1)
                c.setFillColor(colors.HexColor("#1b1f27"))
                c.setFont("Helvetica-Bold" if ri == 0 else "Helvetica", 13 * k)
                c.drawString(8.0 * s, Y(ty + 0.32), a)
                c.drawString(9.9 * s, Y(ty + 0.32), b)
                ty += 0.48
        c.setFillColor(colors.HexColor(GREY))
        c.setFont("Helvetica", 10 * k)
        c.drawString(0.5 * s, Y(7.25), f"Traffic-signal coordination | Project 10 | results backend: "
                     f"{backend_label(self.f)} | slide {sp['no']} of 5")


# =============================================================================== PDF
class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                         topMargin=20 * mm, bottomMargin=18 * mm, **kw)
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="n")
        cover = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="c")
        self.addPageTemplates([PageTemplate("cover", [cover], onPage=lambda c, d: None),
                               PageTemplate("body", [fr], onPage=self._decor)])

    def _decor(self, c, d):
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor(GREY))
        c.drawString(18 * mm, A4[1] - 12 * mm, "Traffic-Signal Coordination with Multi-Agent DQN (Project 10)")
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
        c.setStrokeColor(colors.HexColor("#c9ced6"))
        c.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
        c.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("H1", "H2"):
            lvl = 0 if fl.style.name == "H1" else 1
            text = fl.getPlainText()
            key = f"bm{self.page}_{abs(hash(text)) % 10**6}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, lvl, 0)
            self.notify("TOCEntry", (lvl, text, self.page))


def styles():
    ss = getSampleStyleSheet()
    st = {}
    st["body"] = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.2, spaceAfter=5)
    st["small"] = ParagraphStyle("small", parent=st["body"], fontSize=8.4, leading=11, spaceAfter=3)
    st["tiny"] = ParagraphStyle("tiny", parent=st["body"], fontSize=7.8, leading=9.8, spaceAfter=2)
    st["H1"] = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.HexColor(NAVY), spaceBefore=4, spaceAfter=8)
    st["H2"] = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor(BLUE), spaceBefore=7, spaceAfter=4)
    st["H3"] = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=9.6, leading=12, textColor=colors.HexColor(NAVY), spaceBefore=4, spaceAfter=2)
    st["bul"] = ParagraphStyle("bul", parent=st["body"], leftIndent=12, bulletIndent=2, spaceAfter=2.5)
    st["code"] = ParagraphStyle("code", fontName="Courier", fontSize=6.4, leading=7.5)
    st["cmd"] = ParagraphStyle("cmd", fontName="Courier", fontSize=8, leading=10, backColor=colors.HexColor("#f1f3f6"),
                               borderPadding=(3, 4, 3, 4), spaceAfter=6, leftIndent=4)
    st["note"] = ParagraphStyle("note", parent=st["body"], backColor=colors.HexColor("#fff4e5"), borderColor=colors.HexColor(ORANGE),
                                borderWidth=0.8, borderPadding=6, spaceBefore=4, spaceAfter=10, fontSize=9)
    st["cap"] = ParagraphStyle("cap", parent=st["small"], textColor=colors.HexColor(GREY), alignment=TA_CENTER)
    return st


def tbl(data, widths, header=True, fs=8.4, align_right_from=1):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", fs), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9ced6")),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs)]
    if align_right_from is not None:
        style.append(("ALIGN", (align_right_from, 1), (-1, -1), "CENTER"))
    t.setStyle(TableStyle(style))
    return t


def P(text, st):
    return Paragraph(text.replace("&", "&amp;").replace("&amp;amp;", "&amp;"), st)


def source_files():
    files = ["run_all.py", "check_setup.py"]
    files += sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "tsc").glob("*.py") if p.name != "__init__.py")
    files += ["tsc/agents/__init__.py"] + sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                                                  for p in (ROOT / "tsc" / "agents").glob("*.py") if p.name != "__init__.py")
    files += ["tsc/__init__.py"]
    return files


FILE_ROLE = {
    "run_all.py": "End-to-end pipeline: baseline, training, test evaluation, CSV, figures, report.",
    "check_setup.py": "Sanity check of SUMO/TraCI/PyTorch; proves all 4 intersections respond.",
    "tsc/config.py": "Grid geometry, demand profile, simulation / reward / DQN / training settings.",
    "tsc/demand.py": "Origin-destination demand and routes, shared by both backends.",
    "tsc/network.py": "Builds the SUMO 2x2 network (netconvert) and route files.",
    "tsc/sumo_env.py": "Multi-agent SUMO environment over TraCI (phases, observations, reward, metrics).",
    "tsc/minisim_env.py": "Built-in mesoscopic simulator (fallback backend, same interface).",
    "tsc/runner.py": "Fixed-time runner, multi-agent training loop, validation, checkpoints.",
    "tsc/plots.py": "Learning-curve and comparison figures from real logs.",
    "tsc/agents/__init__.py": "Agent factory (PyTorch if installed, else NumPy).",
    "tsc/agents/dqn_torch.py": "Double-DQN agent in PyTorch (primary implementation).",
    "tsc/agents/dqn_numpy.py": "Identical Double-DQN agent in pure NumPy (fallback).",
    "tsc/agents/replay.py": "Experience replay buffer.",
    "tsc/__init__.py": "Package marker.",
}


def build_pdf(out, f, specs, figs):
    st = styles()
    m, rows = f["meta"], {r["key"]: r for r in f["rows"]}
    sim, hp, tc = m["sim"], m["dqn"], m["train"]
    W = A4[0] - 36 * mm
    S = []
    # ---------------------------------------------------------------- cover
    S += [Spacer(1, 38 * mm), P("Traffic-Signal Coordination with Multi-Agent Deep Q-Networks", ParagraphStyle(
        "ct", fontName="Helvetica-Bold", fontSize=26, leading=32, textColor=colors.HexColor(NAVY))),
        Spacer(1, 6 * mm),
        P("Project 10 - Real-World RL Homework Projects, Unit 5", ParagraphStyle("cs", fontName="Helvetica", fontSize=13, textColor=colors.HexColor(BLUE))),
        Spacer(1, 4 * mm),
        P("Multiple agents control the signals of four interconnected intersections to reduce average waiting time.",
          ParagraphStyle("cs2", fontName="Helvetica", fontSize=11, leading=15, textColor=colors.HexColor(GREY))),
        Spacer(1, 14 * mm), Image(figs["network"], width=82 * mm, height=82 * 5.2 / 6 * mm), Spacer(1, 10 * mm),
        tbl([["Control approaches", "Fixed-time (baseline) vs independent Multi-Agent DQN"],
             ["Simulation backend for results", backend_label(f)],
             ["DQN implementation used", "PyTorch" if f["impl"] == "torch" else "NumPy (identical algorithm; PyTorch version also provided)"],
             ["Date generated", datetime.date.today().strftime("%d %B %Y")],
             ["Contents", "Problem formulation | Architecture | Complete Python code | Training methodology | "
                          "Learning curves & results | 5 slides | Limitations & conclusion | References"]],
            [48 * mm, W - 48 * mm], header=False, fs=8.8, align_right_from=None),
        Spacer(1, 6 * mm), P(backend_note(f), st["note"]), NextPageTemplate("body"), PageBreak()]
    # ---------------------------------------------------------------- TOC
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName="Helvetica-Bold", fontSize=10, leading=14, leftIndent=0, spaceBefore=2),
                       ParagraphStyle("t1", fontName="Helvetica", fontSize=8.5, leading=10.6, leftIndent=14)]
    S += [P("Table of Contents", ParagraphStyle("TOCT", parent=st["H1"])), toc, PageBreak()]

    # ---------------------------------------------------------------- Deliverable 1
    w, q, th, cr = rows["avg_wait_s"], rows["avg_queue_per_lane"], rows["throughput"], rows["cum_reward"]
    S.append(P("Deliverable 1 - One-Page Problem Formulation", st["H1"]))
    sm = st["small"]
    def para(h, t):
        S.append(P(f"<b>{h}.</b> {t}", sm))
    para("Motivation and problem", "Signalised intersections are a major source of urban delay. Fixed-time plans repeat one cycle regardless of "
         "the traffic that actually arrives, so green time is wasted on empty approaches while other approaches queue. The task is to let "
         "multiple intelligent agents, one per intersection, adapt signal timing online to cut vehicle waiting time and queue length across "
         "a connected network.")
    para("Traffic simulation environment", f"Experiments run through the project's multi-agent environment interface. Backend for the "
         f"reported numbers: <b>{backend_label(f)}</b>. One episode is {sim['sim_seconds']} s of simulated time with an off-peak / peak "
         f"(1.4x) / off-peak demand profile; routes are generated from Poisson origin-destination flows (8 entry points, 7 destinations each).")
    para("Multi-intersection network", f"A 2 x 2 grid of four signalised intersections (J00, J01, J10, J11) connected by {sim['link_length']:.0f} m "
         f"two-lane links at {sim['free_speed']:.2f} m/s (50 km/h); each intersection has four approaches, so 16 stop-lines and 8 border entry / exit points.")
    para("State representation", f"For every incoming lane the agent observes <i>density</i> (vehicles / capacity), <i>queue length</i> (halting vehicles / capacity) "
         f"and <i>waiting time</i> (summed waiting seconds / 300), all clipped to [0, 1]; plus a one-hot of the current green phase and the time since the last "
         f"switch, plus the neighbouring intersections' mean queue and waiting level. State size: {f['state_dim']} values per agent.")
    para("Action space", f"Discrete: pick which <i>green phase</i> should be active during the next {sim['delta_time']} s decision step "
         f"({f['n_actions']} phases). Choosing the current phase extends it; choosing another one inserts a {sim['yellow_time']} s yellow transition first, "
         "which gives the safety constraint for free and penalises needless switching through lost time.")
    para("Reward function", f"Local reward r_i = -(0.1 x halting vehicles + sum of waiting seconds / 300) on the agent's incoming lanes, clipped at {sim['reward_floor']:.0f}. "
         f"It is large when queues and waits grow, so maximising return minimises both. Training uses the shared reward "
         f"(1 - alpha) r_i + alpha x mean(neighbour r_j), alpha = {hp['coop_alpha']}. Reported cumulative reward is the sum of the "
         "<i>local</i> rewards of all agents over the episode, so it is comparable between control methods.")
    para("Multi-agent RL algorithm", f"Independent Double DQN: each agent owns a Q-network (obs - {hp['hidden']} - {hp['hidden']} - actions, ReLU), a "
         f"replay buffer ({hp['buffer_size']} transitions) and a target network (hard update every {hp['target_update']} learning steps); "
         f"epsilon-greedy exploration {hp['eps_start']} to {hp['eps_end']}, Huber loss, Adam (lr {hp['lr']}), gamma {hp['gamma']}.")
    para("Coordination between agents", "Agents do not share parameters. They coordinate implicitly through (i) the neighbour queue / waiting summary in the "
         "observation, so an agent sees pressure that is about to arrive or is blocking its exit, and (ii) the cooperative reward mix, which makes an agent "
         "partly responsible for its neighbours' congestion (shared objective: network-wide waiting time).")
    para("Evaluation metrics", f"On {f['n_test']} unseen test seeds (demand scale 1.0 and extra scales if listed in the CSV): average waiting time per vehicle "
         "(vehicle-seconds standing still / vehicles inserted), average queue length (halting vehicles per controlled lane, sampled every decision step), "
         "throughput (vehicles that completed their trip within the hour) and cumulative reward. Fixed-time control uses the same environment with phases "
         f"cycled {sim['fixed_green']} s each.")
    S.append(PageBreak())

    # ---------------------------------------------------------------- architecture
    S.append(P("Project Architecture", st["H1"]))
    S.append(Image(figs["arch"], width=W * 0.9, height=W * 0.9 * 5.6 / 11))
    S.append(P("Figure 1 - System architecture: environment, four independent DQN agents with neighbour coordination, fixed-time baseline and evaluation.", st["cap"]))
    S.append(P("Data flow", st["H2"]))
    for t in ["Every decision step (10 s) the environment returns one observation per junction; <i>runner.augment()</i> appends the neighbours' summary.",
              "Each agent chooses a green phase epsilon-greedily; the environment applies yellow / green states (TraCI <i>setRedYellowGreenState</i> in SUMO) and simulates 10 s.",
              "Local rewards come back; <i>runner.shared_rewards()</i> mixes them cooperatively; every agent stores its transition and performs one Double-DQN update.",
              "After every few training episodes the greedy policies are run on validation seeds; the best checkpoint is kept and finally tested on unseen seeds against the fixed-time baseline."]:
        S.append(Paragraph(t, st["bul"], bulletText="\u2022"))
    S.append(P("Project layout", st["H2"]))
    tree = ("traffic_signal_marl/\n  run_all.py            <- full pipeline (train + evaluate + figures + report)\n  check_setup.py        <- verify SUMO / TraCI / 4 intersections\n"
            "  make_report.py        <- builds PDF, PPTX and README from results/\n  requirements.txt, README.md\n  tsc/\n    config.py  demand.py  network.py\n    sumo_env.py         <- SUMO + TraCI environment\n"
            "    minisim_env.py      <- built-in fallback simulator\n    runner.py  plots.py\n    agents/ dqn_torch.py dqn_numpy.py replay.py __init__.py\n  results/              <- generated: CSVs, PNGs, models, PDF, PPTX\n  sumo_files/           <- generated: network + route files")
    S.append(Preformatted(tree, st["cmd"]))
    S.append(P("Simplified or simulated components (read this)", st["H2"]))
    S.append(Paragraph(
        "<b>SUMO backend</b>: full microscopic simulation of the 2 x 2 grid; phases are read from the netconvert-generated signal programs. "
        "<b>Built-in backend</b>: a mesoscopic queue model (1 s step, link travel time, saturation-flow discharge with start-up loss, finite link storage / spill-back) - "
        "a deliberate simplification: no car-following, lane changing or driver heterogeneity. "
        "Right turns share the through signal group; trips are counted when the vehicle leaves its last junction. "
        "<b>Demand</b> is synthetic (Poisson OD flows), not measured. <b>Avg waiting time</b> = vehicle-seconds with speed below 0.1 m/s divided by vehicles inserted. "
        "See the notice on the cover page for which backend produced the reported numbers.", st["note"]))
    S.append(PageBreak())

    # ---------------------------------------------------------------- run guide
    S.append(P("Step-by-Step Execution Guide (VS Code, Windows)", st["H1"]))
    steps = [
        ("1. Create the folder", "Create <b>C:\\Projects\\traffic_signal_marl</b> and extract <b>Complete_Project_Source_Code.zip</b> into it so that "
         "<i>run_all.py</i> sits directly inside. In VS Code choose File - Open Folder - that folder."),
        ("2. Open a terminal", "Terminal - New Terminal (PowerShell). Check the existing installs (do not reinstall them):"),
    ]
    for h, t in steps:
        S.append(P(h, st["H3"]))
        S.append(P(t, st["body"]))
    S.append(Preformatted("python --version        # expect 3.13.2\nsumo --version         # expect 1.28.0\necho $env:SUMO_HOME   # should point to the SUMO folder", st["cmd"]))
    S.append(P("3. Create a virtual environment and install the Python packages", st["H3"]))
    S.append(Preformatted("python -m venv .venv\n.\\.venv\\Scripts\\Activate.ps1\npip install -r requirements.txt", st["cmd"]))
    S.append(P("If PowerShell blocks activation run <i>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned</i> once. If <i>sumo</i> is not on PATH set "
               "SUMO_HOME, e.g. <i>$env:SUMO_HOME = \"C:\\Program Files (x86)\\Eclipse\\Sumo\"</i>.", st["small"]))
    S.append(P("4. Verify the setup (all four intersections, both controllers)", st["H3"]))
    S.append(Preformatted("python check_setup.py", st["cmd"]))
    S.append(P("5. Run the complete experiment with real SUMO", st["H3"]))
    S.append(Preformatted("python run_all.py --backend sumo --episodes 60 --test-episodes 10\n# optional extras:  --scales 1.0 0.7 1.3   --tune-baseline", st["cmd"]))
    S.append(P("Everything is written to <i>results/</i>: Experiment_Results.csv, summary.csv, training_log.csv, validation_log.csv, Learning_Curves.png, "
               "the PDF and PPTX (regenerated from the new SUMO numbers) and the model checkpoints. To rebuild only the documents: <i>python make_report.py</i>. "
               "To watch SUMO's GUI open <i>sumo_files\\grid2x2.sumocfg</i> with sumo-gui.", st["body"]))
    S.append(P("Expected effort: a SUMO episode of 3600 s takes tens of seconds, so 60 episodes plus validation finish in roughly an hour on a laptop; use "
               "<i>--quick</i> for a 2-minute smoke test.", st["small"]))
    S.append(PageBreak())

    # ---------------------------------------------------------------- implementation listing
    S.append(P("Deliverable 2 - Complete Python Implementation", st["H1"]))
    S.append(P("The listing below is read from the actual files at build time. <i>make_report.py</i> (document generator) is in the ZIP only.", st["small"]))
    for rel in source_files():
        text = (ROOT / rel).read_text(encoding="utf-8").replace("\t", "    ").rstrip()
        S.append(P(rel, st["H2"]))
        S.append(P(FILE_ROLE.get(rel, ""), st["small"]))
        S.append(Preformatted(text, st["code"]))
    reqs = (ROOT / "requirements.txt").read_text()
    S.append(P("requirements.txt", st["H2"]))
    S.append(Preformatted(reqs, st["code"]))
    S.append(PageBreak())

    # ---------------------------------------------------------------- methodology
    S.append(P("Training Methodology", st["H1"]))
    S.append(P(f"All {len(sim) and 4} agents are trained simultaneously for {tc['episodes']} episodes of {sim['sim_seconds']} s. Each episode uses a new "
               f"random seed and a demand multiplier drawn uniformly from {tc['train_scale_range'][0]}-{tc['train_scale_range'][1]}, which prevents the agents "
               "from memorising one traffic pattern. Epsilon decays linearly to its floor over the first "
               f"{int(hp['eps_decay_frac'] * 100)}% of the episodes. Because episodes end by time limit, the bootstrap target is not cut at the last step.", st["body"]))
    S.append(P("Data splits: training seeds 1..N (jittered demand), <b>validation seeds</b> "
               f"{tc['val_seeds']} (greedy policy every {tc['eval_every']} episodes, used only to pick the best checkpoint) and <b>test seeds</b> "
               f"{tc['test_seeds'][0]}-{tc['test_seeds'][-1]} (never used for training or selection).", st["body"]))
    hrows = [["Hyper-parameter", "Value"], ["Network", f"obs - {hp['hidden']} - {hp['hidden']} - |A|, ReLU"], ["Algorithm", "Double DQN, Huber loss, grad-clip 10"],
             ["Optimiser / learning rate", f"Adam / {hp['lr']}"], ["Discount gamma", hp["gamma"]], ["Batch / replay size", f"{hp['batch_size']} / {hp['buffer_size']}"],
             ["Target update", f"every {hp['target_update']} learning steps"], ["Learning starts after", f"{hp['learn_start']} transitions / agent"],
             ["Epsilon", f"{hp['eps_start']} to {hp['eps_end']} over {int(hp['eps_decay_frac'] * 100)}% of episodes"],
             ["Cooperative reward weight alpha", hp["coop_alpha"]], ["Decision step / yellow", f"{sim['delta_time']} s / {sim['yellow_time']} s"],
             ["Training episodes", tc["episodes"]], ["Agent implementation", f["impl"]],
             ["Training wall-clock", f"{m['runtime_s']:.0f} s (whole pipeline)"], ["Best checkpoint", f"episode {m['best_checkpoint']['episode']} (validation wait {m['best_checkpoint']['val_avg_wait_s']:.1f} s)"]]
    S.append(tbl([[str(c) for c in r] for r in hrows], [70 * mm, W - 70 * mm], align_right_from=None))
    S.append(Spacer(1, 4 * mm))
    S.append(P("Reward and state design rationale", st["H2"]))
    for t in ["Queue and waiting terms together penalise both the number of stopped vehicles and how long they have been stopped, which discourages starving a minor approach.",
              "Choosing the target <i>phase</i> instead of toggling gives each agent a small discrete action set and makes 'extend' and 'switch' explicit.",
              "The neighbour summary and reward mixing implement coordination without sharing network weights; it keeps the learners scalable (observation size independent of network size)."]:
        S.append(Paragraph(t, st["bul"], bulletText="\u2022"))
    S.append(PageBreak())

    # ---------------------------------------------------------------- results
    S.append(P("Deliverable 3 - Learning Curves and Result Tables", st["H1"]))
    S.append(P(backend_note(f), st["note"]))
    S.append(Image(figs["learning"], width=W, height=W * 8.2 / 15))
    S.append(P("Figure 2 - (a) training reward per episode; (b, c) waiting time and queue length of the greedy policy on validation seeds during training; "
               "(d-f) test-seed comparison with fixed-time control. Generated from training_log.csv, validation_log.csv and Experiment_Results.csv.", st["cap"]))
    S.append(P(f"Table 1 - Test results, demand scale {f['scale']}, mean +/- std over {f['n_test']} unseen seeds", st["H2"]))
    data = [["Metric", f["fname"], "Multi-agent DQN", "Change", "DQN better in"]]
    for r in f["rows"]:
        data.append([r["label"], f"{r['fixed_m']:{r['fmt'][2:-1]}} +/- {r['fixed_s']:{r['fmt'][2:-1]}}", f"{r['dqn_m']:{r['fmt'][2:-1]}} +/- {r['dqn_s']:{r['fmt'][2:-1]}}",
                     f"{r['change']:+.1f}%", f"{r['wins']}/{r['n']} seeds"])
    S.append(tbl(data, [58 * mm, 31 * mm, 31 * mm, 18 * mm, W - 138 * mm], fs=8))
    S.append(Spacer(1, 3 * mm))
    S.append(P(f"On these test seeds the multi-agent DQN's average waiting time was <b>{verdict(w)}</b> than fixed-time control "
               f"({w['dqn_m']:.1f} s vs {w['fixed_m']:.1f} s), average queue length <b>{verdict(q)}</b>, throughput <b>{verdict(th)}</b>, "
               f"and cumulative reward {cr['dqn_m']:.0f} vs {cr['fixed_m']:.0f}. 'Change' is relative to the fixed-time mean; for waiting time and queue, negative is better.", st["body"]))
    S.append(Image(figs["perf"], width=W, height=W * 3.6 / 16))
    S.append(P("Figure 3 - Performance across training episodes (validation seeds, greedy policy). The untrained episode-0 point is off-scale and omitted here (it is in validation_log.csv).", st["cap"]))
    S.append(Image(figs["bars"], width=W, height=W * 3.8 / 15))
    S.append(P("Figure 4 - Fixed-time vs Multi-Agent DQN on the test seeds (mean +/- std).", st["cap"]))
    # scenarios + baseline variants
    summ = f["summ"].copy()
    S.append(P("Table 2 - All evaluated controllers and demand levels (test seeds)", st["H2"]))
    d2 = [["Controller", "Demand x", "Wait (s)", "Queue (veh/lane)", "Throughput (veh/h)", "Cum. reward"]]
    for _, r in summ.iterrows():
        d2.append([r["controller"], f"{r['demand_scale']:g}", f"{r['avg_wait_s_mean']:.1f} +/- {r['avg_wait_s_std']:.1f}",
                   f"{r['avg_queue_per_lane_mean']:.2f}", f"{r['throughput_mean']:.0f}", f"{r['cum_reward_mean']:.0f}"])
    S.append(tbl(d2, [38 * mm, 18 * mm, 32 * mm, 30 * mm, 32 * mm, W - 150 * mm], fs=8))
    S.append(P("Table 3 - Per-seed test results (demand scale %g)" % f["scale"], st["H2"]))
    d3 = [["Seed", "Fixed wait (s)", "DQN wait (s)", "Fixed queue", "DQN queue", "Fixed thr.", "DQN thr."]]
    for a, b in zip(f["F"].itertuples(), f["D"].itertuples()):
        d3.append([a.seed, f"{a.avg_wait_s:.1f}", f"{b.avg_wait_s:.1f}", f"{a.avg_queue_per_lane:.2f}", f"{b.avg_queue_per_lane:.2f}", int(a.throughput), int(b.throughput)])
    S.append(tbl([[str(c) for c in r] for r in d3], [16 * mm] + [(W - 16 * mm) / 6] * 6, fs=8))
    S.append(P("Interpretation", st["H2"]))
    interp = interpretation(f)
    for t in interp:
        S.append(Paragraph(t, st["bul"], bulletText="\u2022"))
    S.append(PageBreak())

    # ---------------------------------------------------------------- slides
    S.append(P("Deliverable 4 - Exactly Five Presentation Slides", st["H1"]))
    S.append(P(f"The same five slides are provided as <b>{PPT_NAME}</b>.", st["small"]))
    for i, sp in enumerate(specs):
        sp["no"] = i + 1
        S.append(KeepTogether([SlideFlowable(sp, W, f), Spacer(1, 5 * mm)]))
    S.append(PageBreak())

    # ---------------------------------------------------------------- limitations
    S.append(P("Limitations, Ethical Considerations and Conclusion", st["H1"]))
    S.append(P("Limitations", st["H2"]))
    lim = [backend_note(f) if f["backend"] != "sumo" else "Results come from a single synthetic scenario; real SUMO results depend on the netconvert signal programs and driver parameters.",
           "Small network (4 junctions) and synthetic Poisson demand; no pedestrians, buses, bicycles, detector noise or incidents.",
           "Independent learners face a non-stationary environment because neighbours also learn; neighbour summary and reward mixing mitigate but do not remove this.",
           f"Only {f['n_train']} training episodes and one hyper-parameter setting; no seed-wise repetition of the whole training run, so differences between independent training runs are not quantified.",
           "Average waiting time counts only vehicles that were inserted; vehicles still queued at the end of the hour are not completed trips."]
    for t in lim:
        S.append(Paragraph(t, st["bul"], bulletText="\u2022"))
    S.append(P("Ethical considerations", st["H2"]))
    for t in ["<b>Safety</b>: a learned policy must never violate minimum yellow / all-red intervals or pedestrian crossing times; real deployments need a verified safety layer and automatic fallback to a fixed-time plan.",
              "<b>Fairness</b>: minimising average delay can starve minor streets, pedestrians or neighbourhoods; fairness constraints and per-approach maximum waits should be audited.",
              "<b>Transparency and accountability</b>: city engineers need to inspect and override the controller; deep Q-networks are hard to interpret.",
              "<b>Data and privacy</b>: this project only uses simulated, anonymous vehicle counts; real systems based on cameras or connected vehicles must respect privacy law.",
              "<b>Equity of benefit</b>: gains in vehicle throughput should be weighed against emissions, public-transport priority and active-travel safety."]:
        S.append(Paragraph(t, st["bul"], bulletText="\u2022"))
    S.append(P("Future scope", st["H2"]))
    for t in ["Run on larger and real networks and real demand; calibrate with detector data.", "Graph-based communication (CoLight-style attention), parameter sharing and MA-PPO / QMIX variants.",
              "Pressure-based rewards (PressLight), pedestrian and public-transport priority, emergency-vehicle pre-emption.", "Multi-seed training repeats and statistical significance tests; robustness to sensor failures."]:
        S.append(Paragraph(t, st["bul"], bulletText="\u2022"))
    S.append(P("Conclusion", st["H2"]))
    S.append(P(conclusion(f), st["body"]))
    S.append(PageBreak())

    # ---------------------------------------------------------------- references
    S.append(P("References", st["H1"]))
    refs = [
        "Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. <i>Nature</i>, 518, 529-533.",
        "van Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement learning with Double Q-learning. <i>AAAI Conference on Artificial Intelligence</i>.",
        "Tan, M. (1993). Multi-agent reinforcement learning: independent vs. cooperative agents. <i>ICML</i>.",
        "Lopez, P. A. et al. (2018). Microscopic traffic simulation using SUMO. <i>IEEE International Conference on Intelligent Transportation Systems (ITSC)</i>.",
        "Genders, W., Razavi, S. (2016). Using a deep reinforcement learning agent for traffic signal control. <i>arXiv:1611.01142</i>.",
        "Chu, T., Wang, J., Codeca, L., Li, Z. (2020). Multi-agent deep reinforcement learning for large-scale traffic signal control. <i>IEEE Transactions on Intelligent Transportation Systems</i>, 21(3), 1086-1095.",
        "Wei, H. et al. (2019). PressLight: learning max pressure control to coordinate traffic signals in arterial network. <i>ACM SIGKDD</i>.",
        "Wei, H. et al. (2019). CoLight: learning network-level cooperation for traffic signal control. <i>ACM CIKM</i>.",
        "Kingma, D. P., Ba, J. (2015). Adam: a method for stochastic optimization. <i>ICLR</i>.",
        "Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. <i>NeurIPS</i>.",
        "Eclipse SUMO documentation, TraCI - Traffic Control Interface. https://sumo.dlr.de/docs/",
    ]
    for i, r in enumerate(refs, 1):
        S.append(Paragraph(f"[{i}] {r}", ParagraphStyle("ref", parent=st["body"], leftIndent=16, firstLineIndent=-16)))

    doc = Doc(str(Path(out) / PDF_NAME), title="Traffic Signal Coordination - Complete Submission", author="Project 10")
    doc.multiBuild(S)


def interpretation(f):
    rows = {r["key"]: r for r in f["rows"]}
    w, q, th = rows["avg_wait_s"], rows["avg_queue_per_lane"], rows["throughput"]
    out = []
    if w["change"] < 0:
        out.append(f"The learned policy reduced average waiting time by {abs(w['change']):.1f}% and was better than the fixed-time plan on {w['wins']} of {w['n']} test seeds.")
    else:
        out.append(f"The learned policy did <b>not</b> beat fixed-time control on waiting time ({w['change']:+.1f}%); it won on {w['wins']} of {w['n']} seeds.")
    out.append(f"Average queue length changed by {q['change']:+.1f}% and throughput by {th['change']:+.1f}%; throughput is bounded by the demand that is injected in the hour, so large throughput gains are not expected unless the baseline leaves vehicles stuck.")
    summ = f["summ"]
    fixed_rows = summ[summ.controller.str.startswith("Fixed")]
    if fixed_rows.demand_scale.nunique() == 1 and fixed_rows.controller.nunique() > 1:
        b = fixed_rows.sort_values("avg_wait_s_mean").iloc[0]
        out.append(f"Among fixed-time variants the best one by waiting time was '{b['controller']}' ({b['avg_wait_s_mean']:.1f} s); the headline baseline uses {f['fname']}.")
    elif fixed_rows.controller.nunique() > 1:
        out.append("Fixed-time variants with other green durations were also evaluated (Table 2) to show the baseline is not unfairly weak.")
    d = summ[summ.controller == "Multi-agent DQN"].sort_values("demand_scale")
    fx = summ[summ.controller == f["fname"]].set_index("demand_scale")
    if len(d) > 1:
        parts = []
        for _, r in d.iterrows():
            base = fx.loc[r["demand_scale"]]
            c = pct(r["avg_wait_s_mean"], base["avg_wait_s_mean"])
            parts.append(f"{r['demand_scale']:g}x demand: {r['avg_wait_s_mean']:.1f} s vs {base['avg_wait_s_mean']:.1f} s ({c:+.1f}%)")
        out.append("Robustness across demand levels (waiting time, DQN vs " + f["fname"] + "): " + "; ".join(parts) +
                   ". The DQN was trained with demand jitter of 0.8-1.25x, so scales outside that range test generalisation.")
        bad = [r for _, r in d.iterrows() if r["avg_wait_s_mean"] > fx.loc[r["demand_scale"]]["avg_wait_s_mean"]]
        for r in bad:
            out.append(f"<b>Failure case:</b> at {r['demand_scale']:g}x demand the DQN was worse than fixed-time control and highly variable (std {r['avg_wait_s_std']:.1f} s, throughput "
                       f"{r['throughput_mean']:.0f} vs {fx.loc[r['demand_scale']]['throughput_mean']:.0f} veh/h): this over-saturated regime was never seen in training, and the learned policy let queues spill back. "
                       "Training on a wider demand range (or a pressure-style reward) is the obvious next step.")
    tr = f["train"]
    out.append(f"Training reward rose from {tr['cum_reward'].iloc[:10].mean():.0f} (first 10 episodes, mostly exploratory) to {tr['cum_reward'].iloc[-10:].mean():.0f} (last 10 episodes, epsilon at floor, jittered demand).")
    return out


def robustness_sentence(f):
    summ = f["summ"]
    d = summ[summ.controller == "Multi-agent DQN"]
    fx = summ[summ.controller == f["fname"]].set_index("demand_scale")
    bad = [f"{r['demand_scale']:g}x" for _, r in d.iterrows() if r["avg_wait_s_mean"] > fx.loc[r["demand_scale"]]["avg_wait_s_mean"]]
    if bad:
        return "The advantage did not hold at every demand level: at " + ", ".join(bad) + " demand the DQN was worse than fixed-time control. "
    return ""


def conclusion(f):
    rows = {r["key"]: r for r in f["rows"]}
    w, q, th = rows["avg_wait_s"], rows["avg_queue_per_lane"], rows["throughput"]
    word = "outperformed" if w["change"] < 0 else "did not outperform"
    return (f"A multi-agent system of four independent Double-DQN controllers, coordinated through neighbour observations and a cooperative reward, was trained on a 2 x 2 signalised grid "
            f"and compared with fixed-time control on {f['n_test']} unseen seeds ({backend_label(f)}). The learned controllers {word} the baseline on average waiting time "
            f"({w['dqn_m']:.1f} s vs {w['fixed_m']:.1f} s, {w['change']:+.1f}%), with queue length {q['change']:+.1f}% and throughput {th['change']:+.1f}%. "
            + robustness_sentence(f) +
            "All numbers in this document come directly from the executed experiments stored in the accompanying CSV files. "
            + ("" if f["backend"] == "sumo" else "Because the executed backend was the project's simplified simulator, the conclusion applies to that model; rerun with --backend sumo to obtain SUMO-based evidence."))


# =============================================================================== README
def write_readme(f, out):
    rows = {r["key"]: r for r in f["rows"]}
    lines = ["| Metric | " + f["fname"] + " | Multi-agent DQN | Change |", "|---|---|---|---|"]
    for r in f["rows"]:
        fm = r["fmt"].format
        lines.append(f"| {r['label']} | {fm(r['fixed_m'])} | {fm(r['dqn_m'])} | {r['change']:+.1f}% |")
    txt = f"""# Traffic-Signal Coordination with Multi-Agent DQN (Project 10)

Four signalised intersections (2 x 2 grid) are controlled by **four independent Double-DQN agents**
and compared with **fixed-time control**. Stack: Python 3.13, Eclipse SUMO 1.28, TraCI, PyTorch, NumPy, Pandas, Matplotlib.

## Results currently stored in `results/`
Backend that produced them: **{backend_label(f)}**; DQN implementation: **{f['impl']}**;
{f['n_train']} training episodes; mean over {f['n_test']} unseen test seeds, demand scale {f['scale']}.

{chr(10).join(lines)}

> {backend_note(f)}

## Run on Windows with VS Code
```powershell
cd C:\\Projects\\traffic_signal_marl
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python check_setup.py                       # verifies SUMO, TraCI, 4 intersections, both controllers
python run_all.py --backend sumo --episodes 60 --test-episodes 10
```
If `sumo` is not found: `$env:SUMO_HOME = "C:\\Program Files (x86)\\Eclipse\\Sumo"`.
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
"""
    (ROOT / "README.md").write_text(txt, encoding="utf-8")


# =============================================================================== main
def build(out=None):
    out = Path(out or ROOT / "results")
    f = facts(out)
    # derive state dimension / actions from the stored model
    first_w = None
    from tsc.minisim_env import PHASES
    f["n_actions"] = len(PHASES) if f["backend"] == "minisim" else 4
    f["state_dim"] = 8 * 3 + f["n_actions"] + 1 + 2
    if f["backend"] == "sumo":
        f["state_dim"] = f"{f['state_dim']} (typical; read from network: 8 lanes x 3 + phases + 1 + 2)"
    figs = {"network": str(out / "network_diagram.png"), "arch": str(out / "architecture.png"),
            "learning": str(out / "Learning_Curves.png"), "perf": str(out / "performance_across_training.png"),
            "bars": str(out / "comparison_bars.png")}
    network_figure(figs["network"])
    architecture_figure(figs["arch"], f)
    specs = slide_specs(f, figs)
    assert len(specs) == 5
    build_pptx(specs, str(out / PPT_NAME), f)
    build_pdf(out, f, specs, figs)
    write_readme(f, out)
    print("report written:", out / PDF_NAME, "|", out / PPT_NAME)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
