"""Markdown report + charts for the heavy-workload run (experiments/perf.py).

    python -m experiments.perf_report experiments/results/perf.csv --out experiments/results/perf_report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analysis import md_table

SIZE_ORDER = ["XS", "S", "M", "L", "XL"]
METHOD_ORDER = ["B7", "B1", "B2", "B3", "B4", "B5", "B6"]
METHOD_LABEL = {"B1": "B1 local check", "B2": "B2 hard-only chain", "B3": "B3 full (hard only)",
                "B4": "B4 full (regulation)", "B5": "B5 unweighted expansion",
                "B6": "B6 local search", "B7": "B7 proposed"}
# reference categorical palette (light mode), fixed order; B7 takes slot 1
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLOR = dict(zip(METHOD_ORDER, SLOTS))
MARKERS = dict(zip(METHOD_ORDER, ["o", "s", "^", "D", "v", "P", "X"]))
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OUTCOME_GROUPS = [
    ("solved", ["solved"]),
    ("physical infeasibility", ["physical_infeasibility"]),
    ("regulatory infeasibility", ["regulatory_infeasibility"]),
    ("lock conflict", ["lock_conflict"]),
    ("rejected", ["rejected_local_check", "rejected_by_approver"]),
    ("time limit", ["algorithmic_failure", "killed_timeout"]),
    ("memory limit", ["memory_limit"]),
    ("error", ["error", "crashed"]),
]
from experiments.scenarios import SINGLE  # noqa: E402  (single-request scenarios)

# ----------------------------------------------------------- chart language
# Chart texts in Armenian ("hy", default) or English ("en").  Method codes follow
# the paper: Բ1-Բ7 in Armenian, B1-B7 in English.  Report tables stay in English.
LANG = "hy"
TEXTS = {
    "en": {
        "elapsed_title": "Elapsed (wall-clock) time per request, median by size",
        "elapsed_y": "seconds (log scale)",
        "limit": "{v:g} s limit",
        "lessons": "{n} lessons",
        "mem_commit": "Peak committed memory (virtual, private bytes)",
        "mem_wset": "Peak working set (physical RAM)",
        "mem_y": "MB, median per request",
        "guard": "memory guard {v:g} MB",
        "pk_title": "Changes relative to the optimum: P_k / P_k*\n(single-request scenarios, solved)",
        "pk_y": "P_k / P_k*   (1 = minimal changes)",
        "pk_empty": "no solved single-request case with a proven reference yet",
        "out_title": "Outcome mix per size and method",
        "out_y": "share of requests",
    },
    "hy": {
        "elapsed_title": "Հարցման մշակման իրական ժամանակը (մեդիան՝ ըստ խնդրի չափի)",
        "elapsed_y": "վայրկյան (լոգարիթմական սանդղակ)",
        "limit": "{v:g} վ սահման",
        "lessons": "{n} դաս",
        "mem_commit": "Հատկացված վիրտուալ հիշողության առավելագույն ծավալը",
        "mem_wset": "Ֆիզիկական հիշողության առավելագույն ծավալը",
        "mem_y": "ՄԲ, մեդիան՝ մեկ հարցման համար",
        "guard": "հիշողության սահման՝ {v:g} ՄԲ",
        "pk_title": "Փոփոխությունները օպտիմումի համեմատ՝ P_k / P_k*\n(մեկ հարցմամբ լուծված սցենարներ)",
        "pk_y": "P_k / P_k*   (1 = նվազագույն)",
        "pk_empty": "Ապացուցված օպտիմումով լուծված մեկ հարցման դեպք դեռ չկա",
        "out_title": "Ելքերի բաշխումը՝ ըստ խնդրի չափի և մեթոդի",
        "out_y": "հարցումների բաժինը",
    },
}
METHOD_LABEL_HY = {"B1": "Բ1 տեղային ստուգում", "B2": "Բ2 շղթա՝ միայն խիստ պայմաններով",
                   "B3": "Բ3 ամբողջական (միայն խիստ)", "B4": "Բ4 ամբողջական (կանոնակարգով)",
                   "B5": "Բ5 կշռազուրկ ընդլայնում", "B6": "Բ6 տեղային որոնում",
                   "B7": "Բ7 առաջարկվող"}
OUTCOME_LABEL_HY = {"solved": "լուծում", "physical infeasibility": "ֆիզիկական անիրագործելիություն",
                    "regulatory infeasibility": "կանոնակարգային անիրագործելիություն",
                    "lock conflict": "ամրագրման հակասություն", "rejected": "մերժում",
                    "time limit": "ալգորիթմական ձախողում (ժամանակ)",
                    "memory limit": "հիշողության սահման", "error": "սխալ"}


def T(key: str, **fmt) -> str:
    return TEXTS[LANG][key].format(**fmt)


def method_label(m: str) -> str:
    return (METHOD_LABEL_HY if LANG == "hy" else METHOD_LABEL).get(m, m)


def method_code(m: str) -> str:
    return method_label(m).split(" ", 1)[0] if m in METHOD_LABEL else m


def outcome_label(g: str) -> str:
    return OUTCOME_LABEL_HY.get(g, g) if LANG == "hy" else g


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if df[c].dtype == object and c not in ("size", "scenario", "method", "kind", "outcome",
                                                "lexmin_stages", "extra_expansion", "error",
                                                "pstar_proven", "lexmin_proven", "mus_minimal",
                                                "requires_approval", "voluntary"):
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    for c in ("pstar_proven", "lexmin_proven", "mus_minimal", "requires_approval", "voluntary"):
        if c in df:
            df[c] = df[c].map({True: True, False: False, "True": True, "False": False})
    df["outcome_group"] = df["outcome"].map({o: g for g, os in OUTCOME_GROUPS for o in os}).fillna("error")
    return attach_reference(df)


def attach_reference(df: pd.DataFrame) -> pd.DataFrame:
    """P_k* = proven optimal P* of B4 (full re-optimisation under the same
    regulation) on the first request of a single-request scenario.  Older CSVs
    with separate "REF" rows (B4 with a longer limit) use those instead."""
    base = df[(df["step"] == 0) & (df["outcome"] == "solved") & (df["pstar_proven"] == True)]  # noqa: E712
    ref = base[base["method"] == "REF"]
    b4 = base[(base["method"] == "B4")
              & ~base.set_index(["size", "seed", "scenario"]).index.isin(
                  ref.set_index(["size", "seed", "scenario"]).index)]
    ref = pd.concat([ref, b4])[["size", "seed", "scenario", "pstar"]].rename(columns={"pstar": "pk_star"})
    ref = ref.drop_duplicates(["size", "seed", "scenario"])
    out = df.merge(ref, on=["size", "seed", "scenario"], how="left")
    single = out["scenario"].isin(SINGLE) & (out["step"] == 0) & (out["outcome"] == "solved")
    out.loc[~single, "pk_star"] = np.nan
    pk, star = out["P_k"].astype(float), out["pk_star"].astype(float)
    out["pk_excess"] = pk - star
    out["pk_ratio"] = np.where(star > 0, pk / star.replace(0, np.nan), np.where(pk == 0, 1.0, np.nan))
    out.loc[star.isna(), "pk_ratio"] = np.nan
    return out


def _ordered(index, order):
    return [x for x in order if x in set(index)] + sorted(set(index) - set(order))


BASE_SIZES = ["XS", "S", "M", "L", "XL"]


def size_order(df: pd.DataFrame) -> list[str]:
    """Synthetic sizes first, then ITC(-style) instances by number of lessons."""
    known = [s for s in BASE_SIZES if s in set(df["size"])]
    rest = df[~df["size"].isin(BASE_SIZES)].groupby("size")["lessons"].max().sort_values()
    return known + list(rest.index)


# --------------------------------------------------------------------- charts
# Drawn at print size for a 16 cm text width (fonts 7-7.5 pt at 100 %), and
# readable in black and white: every method has its own line style AND marker
# shape (filled / hollow), every outcome its own fill pattern.  No built-in
# figure titles - the paper's captions name the figures; panels carry (a)/(b).
PRINT_WIDTH_IN = 16 / 2.54
DPI = 300
RC = {
    "font.family": "DejaVu Sans",          # has the Armenian glyphs
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "hatch.linewidth": 0.6, "axes.linewidth": 0.6,
    "savefig.dpi": DPI, "figure.dpi": 100,
}
# line style, marker, filled marker?, line width
METHOD_STYLE = {
    "B7": ("-", "o", True, 2.0),
    "B4": ((0, (6, 2)), "s", True, 1.1),
    "B5": ((0, (5, 1.5, 1, 1.5)), "^", True, 1.1),
    "B6": ((0, (2.5, 1.2)), "X", True, 1.1),
    "B1": ((0, (1, 1.5)), "D", False, 1.1),
    "B2": ((0, (4, 1.2, 1, 1.2, 1, 1.2)), "v", False, 1.1),
    "B3": ((0, (9, 2.5)), "s", False, 1.1),
}
REFLINE = dict(color="#c3c2b7", linewidth=2.6, solid_capstyle="butt", zorder=0)
# chart outcome groups: (key, raw outcomes, fill, hatch, hatch colour)
CHART_OUTCOMES = [
    ("solved_proven", None, "#2a78d6", "", None),
    ("solved_cut", None, "#2a78d6", "////", "#ffffff"),
    ("physical infeasibility", ["physical_infeasibility"], "#eb6834", "\\\\\\\\", "#3d3c39"),
    ("regulatory infeasibility", ["regulatory_infeasibility"], "#eda100", "xxx", "#3d3c39"),
    ("lock conflict", ["lock_conflict"], "#6250d6", "...", "#ffffff"),
    ("rejected", ["rejected_local_check", "rejected_by_approver"], "#c3c2b7", "---", "#3d3c39"),
    ("time limit", ["algorithmic_failure", "killed_timeout"], "#3d3c39", "", None),
    ("memory limit", ["memory_limit"], "#e34948", "+++", "#ffffff"),
    ("error", ["error", "crashed"], "#e87ba4", "ooo", "#3d3c39"),
]
CHART_TEXTS = {
    "en": {
        "solved_proven": "solved, every objective stage proven optimal",
        "solved_cut": "solved, stopped by the time limit",
        "synthetic": "synthetic instances", "itc": "instances adapted from ITC 2019",
        "elapsed_y": "seconds per request\n(log scale)",
        "limit": "{v:g} s search limit",
        "mem_commit_y": "committed\nmemory, MB",
        "mem_wset_y": "physical\nmemory, MB",
        "guard": "memory guard {v:g} MB",
        "pk_y": "$P_k\\,/\\,P_k^*$   (1 = fewest changes)",
        "pk_median": "median",
        "pk_empty": "no solved single-request case with a proven reference yet",
        "out_y": "share of requests",
        "lessons": "{n} lessons",
        "size_x": "instance (number of lessons)",
    },
    "hy": {
        "solved_proven": "լուծում՝ բոլոր փուլերում ապացուցված օպտիմալ",
        "solved_cut": "լուծում՝ ընդհատված ժամանակի սահմանով",
        "synthetic": "սինթետիկ օրինակներ", "itc": "ITC 2019-ից հարմարեցված օրինակներ",
        "elapsed_y": "վայրկյան մեկ հայտի համար\n(լոգարիթմական սանդղակ)",
        "limit": "որոնման սահմանը՝ {v:g} վ",
        "mem_commit_y": "հատկացված\nհիշողություն, ՄԲ",
        "mem_wset_y": "ֆիզիկական\nհիշողություն, ՄԲ",
        "guard": "հիշողության սահմանը՝ {v:g} ՄԲ",
        "pk_y": "$P_k\\,/\\,P_k^*$   (1 = նվազագույն)",
        "pk_median": "մեդիան",
        "pk_empty": "Ապացուցված օպտիմումով լուծված մեկ հայտի դեպք դեռ չկա",
        "out_y": "հայտերի բաժինը",
        "lessons": "{n} դաս",
        "size_x": "օրինակ (դասերի քանակ)",
    },
}
PANEL_LETTERS = {"hy": "աբգդեզէ", "en": "abcdefg"}


def CT(key: str, **fmt) -> str:
    return CHART_TEXTS[LANG][key].format(**fmt)


def chart_outcome_label(key: str) -> str:
    if key in ("solved_proven", "solved_cut"):
        return CT(key)
    return outcome_label(key)


def chart_outcome_group(df: pd.DataFrame) -> pd.Series:
    raw = {o: g for g, os, *_ in CHART_OUTCOMES if os for o in os}
    grp = df["outcome"].map(raw).fillna("error")
    solved = df["outcome"] == "solved"
    grp[solved] = np.where(df.loc[solved, "lexmin_proven"] == True, "solved_proven", "solved_cut")  # noqa: E712
    return grp


def _panel(ax, letter_i: int, name: str) -> None:
    ax.set_title(f"{PANEL_LETTERS[LANG][letter_i]}) {name}", loc="left", color=INK, pad=4)


def _axes_style(ax, ylabel: str = "") -> None:
    ax.set_facecolor("#ffffff")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, length=2, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _plot_method(ax, x, y, m, connect=True):
    ls, mk, filled, lw = METHOD_STYLE.get(m, ("-", "o", True, 1.1))
    c = METHOD_COLOR.get(m, MUTED)
    return ax.plot(x, y, color=c, linestyle=ls if connect else "none", linewidth=lw, marker=mk,
                   markersize=4.6 if m == "B7" else 4.0,
                   markerfacecolor=c if filled else "#ffffff", markeredgecolor="#ffffff" if filled else c,
                   markeredgewidth=0.5 if filled else 1.0, label=method_label(m), zorder=3 if m == "B7" else 2,
                   dash_capstyle="butt")[0]


def _instance_groups(d: pd.DataFrame) -> list[tuple[str, list[str], bool]]:
    """(panel name, sizes, connect points by lines) - synthetic sizes, then ITC instances."""
    sizes = _ordered(d["size"].unique(), SIZE_ORDER)
    syn = [s for s in sizes if not s.startswith("itc")]
    itc = [s for s in sizes if s.startswith("itc")]
    return [(n, s, c) for n, s, c in ((CT("synthetic"), syn, True), (CT("itc"), itc, False)) if s]


def _size_label(d: pd.DataFrame, s: str) -> str:
    """Instance name (long ITC names wrapped at their hyphens) and, in brackets,
    the number of lessons; the unit is named once by the x-axis title."""
    n = int(d[d["size"] == s]["lessons"].max())
    name = s.removeprefix("itc-")
    if len(name) > 9:
        name = "-\n".join(name.split("-"))
    return f"{name}\n({n})"


def _outside_legend(fig, handles, labels, ncol: int):
    return fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False,
                      labelcolor=INK2, handlelength=3.2, columnspacing=1.4, handletextpad=0.6)


def text_overlaps(fig) -> list[str]:
    """Every pair of overlapping texts (ticks, axis labels, panel titles,
    legend entries) and every text or legend outside the figure or on a plot."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []

    def add(t, where):
        if t.get_visible() and t.get_text().strip():
            items.append((where, t.get_text().replace("\n", " "), t.get_window_extent(r)))
    visible = [ax for ax in fig.axes if ax.get_visible()]
    for ax in visible:
        add(ax.title, "title")
        add(ax.xaxis.label, "xlabel")
        add(ax.yaxis.label, "ylabel")
        ab = ax.get_window_extent(r)
        for t in ax.get_xticklabels():          # only ticks inside the view are drawn
            c = t.get_window_extent(r)
            if ab.x0 - 1 <= (c.x0 + c.x1) / 2 <= ab.x1 + 1:
                add(t, "tick")
        for t in ax.get_yticklabels():
            c = t.get_window_extent(r)
            if ab.y0 - 1 <= (c.y0 + c.y1) / 2 <= ab.y1 + 1:
                add(t, "tick")
        for t in ax.texts:
            add(t, "text")
    legends = list(fig.legends) + [ax.get_legend() for ax in visible if ax.get_legend()]
    for lg in legends:
        for t in lg.get_texts():
            add(t, "legend")
    W, H = fig.bbox.width, fig.bbox.height
    out = []
    for i, (w1, s1, b1) in enumerate(items):
        if b1.x0 < -0.5 or b1.y0 < -0.5 or b1.x1 > W + 0.5 or b1.y1 > H + 0.5:
            out.append(f"outside the figure: {w1} {s1!r}")
        for w2, s2, b2 in items[i + 1:]:
            dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if dx > 0.5 and dy > 0.5:
                out.append(f"overlap: {w1} {s1!r} / {w2} {s2!r}")
    for lg in fig.legends:
        lb = lg.get_window_extent(r)
        for ax in visible:
            ab = ax.get_tightbbox(r)
            if ab is None:
                continue
            if min(lb.x1, ab.x1) - max(lb.x0, ab.x0) > 0.5 and min(lb.y1, ab.y1) - max(lb.y0, ab.y0) > 0.5:
                out.append("legend overlaps a panel")
    return out


CHART_PROBLEMS: dict[str, list[str]] = {}


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    probs = text_overlaps(fig)
    if probs:
        CHART_PROBLEMS[path.name] = probs
    fig.savefig(path, dpi=DPI, facecolor="#ffffff")
    plt.close(fig)


def chart_elapsed(df: pd.DataFrame, path: Path, limits=(30,)) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    d = df[df["method"].isin(METHOD_ORDER)]
    groups = _instance_groups(d)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(groups), figsize=(PRINT_WIDTH_IN, 3.6), sharey=True,
                                 layout="constrained", squeeze=False,
                                 gridspec_kw=dict(width_ratios=[max(4, len(s)) for _, s, _ in groups]))
        methods = _ordered(d["method"].unique(), METHOD_ORDER)
        dodge = dict(zip(methods, np.linspace(-0.22, 0.22, len(methods))))
        handles = {}
        for i, (ax, (name, sizes, connect)) in enumerate(zip(axes[0], groups)):
            _axes_style(ax, CT("elapsed_y") if i == 0 else "")
            _panel(ax, i, name)
            x = np.arange(len(sizes))
            for lim in limits:
                ref = ax.axhline(lim, **REFLINE)
            for m in methods:
                med = d[d["method"] == m].groupby("size")["elapsed_s"].median().reindex(sizes)
                h = _plot_method(ax, x + (0 if connect else dodge[m]), med.values, m, connect)
                if connect or m not in handles:        # legend: line style + marker
                    handles[m] = h
            ax.set_xticks(x, [_size_label(d, s) for s in sizes])
            ax.set_xlabel(CT("size_x"), color=INK2)
            ax.set_xlim(-0.5, len(sizes) - 0.5)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        hs = [handles[m] for m in methods] + [ref]
        ls = [method_label(m) for m in methods] + [CT("limit", v=limits[0])]
        _outside_legend(fig, hs, ls, ncol=2)
        _save(fig, path)


def chart_memory(df: pd.DataFrame, path: Path, guard_mb: float = 3072) -> None:
    import matplotlib.pyplot as plt
    d = df[df["method"].isin(METHOD_ORDER)]
    groups = _instance_groups(d)
    rows = [("peak_commit_mb", CT("mem_commit_y")), ("peak_wset_mb", CT("mem_wset_y"))]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, len(groups), figsize=(PRINT_WIDTH_IN, 5.2), sharey="row",
                                 layout="constrained", squeeze=False,
                                 gridspec_kw=dict(width_ratios=[max(4, len(s)) for _, s, _ in groups]))
        methods = _ordered(d["method"].unique(), METHOD_ORDER)
        dodge = dict(zip(methods, np.linspace(-0.22, 0.22, len(methods))))
        handles, guard = {}, None
        for r, (col, ylabel) in enumerate(rows):
            for i, (name, sizes, connect) in enumerate(groups):
                ax = axes[r][i]
                _axes_style(ax, ylabel if i == 0 else "")
                if r == 0:
                    _panel(ax, i, name)
                x = np.arange(len(sizes))
                for m in methods:
                    med = d[d["method"] == m].groupby("size")[col].median().reindex(sizes)
                    h = _plot_method(ax, x + (0 if connect else dodge[m]), med.values, m, connect)
                    if connect or m not in handles:    # legend: line style + marker
                        handles[m] = h
                ax.set_xticks(x, [_size_label(d, s) for s in sizes] if r == 1 else [""] * len(sizes))
                if r == 1:
                    ax.set_xlabel(CT("size_x"), color=INK2)
                ax.set_xlim(-0.5, len(sizes) - 0.5)
                ax.set_ylim(bottom=0)
                if col == "peak_commit_mb":
                    guard = ax.axhline(guard_mb, **REFLINE)
        top = max(guard_mb, float(np.nanmax(d["peak_commit_mb"].values)) if len(d) else 0) * 1.08
        axes[0][0].set_ylim(0, top)
        hs = [handles[m] for m in methods] + ([guard] if guard is not None else [])
        ls = [method_label(m) for m in methods] + ([CT("guard", v=guard_mb)] if guard is not None else [])
        _outside_legend(fig, hs, ls, ncol=2)
        _save(fig, path)


def chart_pk_ratio(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = df[df["method"].isin(METHOD_ORDER) & df["pk_ratio"].notna()]
    methods = _ordered(d["method"].unique(), METHOD_ORDER)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN, 2.7), layout="constrained")
        _axes_style(ax, CT("pk_y"))
        if not methods:
            ax.text(0.5, 0.5, CT("pk_empty"), transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_xticks([])
            _save(fig, path)
            return
        ax.axhline(1.0, **REFLINE)
        for i, m in enumerate(methods):
            vals = np.sort(d[d["method"] == m]["pk_ratio"].values)
            # equal values side by side (a small beeswarm), so every case is visible
            xs = np.empty(len(vals))
            for v in np.unique(np.round(vals, 6)):
                idx = np.where(np.round(vals, 6) == v)[0]
                xs[idx] = i + (np.arange(len(idx)) - (len(idx) - 1) / 2) * 0.07
            _plot_method(ax, xs, vals, m, connect=False)
            med = float(np.median(vals))
            ax.plot([i - 0.32, i + 0.32], [med, med], color=INK, linewidth=1.0, zorder=4)
        ax.set_xticks(range(len(methods)), [f"{method_code(m)}\nn = {int((d['method'] == m).sum())}"
                                            for m in methods])
        for t, m in zip(ax.get_xticklabels(), methods):
            if m == "B7":
                t.set_fontweight("bold")
        ax.set_xlim(-0.6, len(methods) - 0.4)
        hs = [Line2D([], [], color=INK, linewidth=1.0)]
        # outside the plot: an in-axes legend could cover data points
        _outside_legend(fig, hs, [CT("pk_median")], ncol=1)
        _save(fig, path)


def chart_outcomes(df: pd.DataFrame, path: Path, which: str = "synthetic") -> bool:
    """Stacked outcome shares per method, one panel per instance; ``which`` is
    "synthetic" (S-XL) or "itc".  Returns False when there is nothing to draw."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    d = df[df["method"].isin(METHOD_ORDER)].copy()
    d["chart_group"] = chart_outcome_group(d)
    sizes = [s for s in _ordered(d["size"].unique(), SIZE_ORDER) if s.startswith("itc") == (which == "itc")]
    if not sizes:
        return False
    ncols = min(4, len(sizes)) if which == "synthetic" else min(3, len(sizes))
    nrows = int(np.ceil(len(sizes) / ncols))
    present = [g for g, *_ in CHART_OUTCOMES if g in set(d["chart_group"])]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(PRINT_WIDTH_IN, 0.6 + 2.0 * nrows + 0.5),
                                 sharey=True, layout="constrained", squeeze=False)
        for k, ax in enumerate(axes.flat):
            if k >= len(sizes):
                ax.set_visible(False)
                continue
            size = sizes[k]
            sub = d[d["size"] == size]
            methods = _ordered(sub["method"].unique(), METHOD_ORDER)
            share = pd.crosstab(sub["method"], sub["chart_group"], normalize="index").reindex(methods)
            _axes_style(ax, CT("out_y") if k % ncols == 0 else "")
            n = int(sub["lessons"].max())
            ax.set_title(f"{PANEL_LETTERS[LANG][k]}) {size.removeprefix('itc-')}, {CT('lessons', n=n)}",
                         loc="left", color=INK, pad=4)
            bottom = np.zeros(len(methods))
            for g, _, color, hatch, hcolor in CHART_OUTCOMES:
                if g not in share:
                    continue
                vals = share[g].fillna(0).values
                ax.bar(range(len(methods)), vals, bottom=bottom, color=color, width=0.74,
                       edgecolor="#ffffff", linewidth=0.7, hatch=hatch or None,
                       hatchcolor=hcolor or color)
                bottom += vals
            ax.set_xticks(range(len(methods)), [method_code(m) for m in methods])
            for t, m in zip(ax.get_xticklabels(), methods):
                if m == "B7":
                    t.set_fontweight("bold")
            ax.set_ylim(0, 1.0)
            ax.set_xlim(-0.6, len(methods) - 0.4)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}".replace("%", " %"))
            ax.tick_params(axis="x", length=0)
        spec = {g: (c, h, hc) for g, _, c, h, hc in CHART_OUTCOMES}
        handles = [Patch(facecolor=spec[g][0], edgecolor="#ffffff", linewidth=0.7, hatch=spec[g][1] or None,
                         hatchcolor=spec[g][2] or spec[g][0]) for g in present]
        _outside_legend(fig, handles, [chart_outcome_label(g) for g in present], ncol=2)
        _save(fig, path)
    return True


# --------------------------------------------------------------------- tables
def _q(p):
    def f(s):
        return s.quantile(p)
    f.__name__ = f"p{int(p * 100)}"
    return f


def tables(df: pd.DataFrame) -> list[str]:
    d = df[df["method"].isin(METHOD_ORDER)].copy()
    d["Q_O_delta"] = d["Q_O"] - d["Q_O_k"]
    solved = d[d["outcome"] == "solved"]
    out = []
    keys = ["size", "method"]

    def sort(t):
        t = t.reset_index()
        t["_s"] = t["size"].map({s: i for i, s in enumerate(SIZE_ORDER)})
        t["_m"] = t["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
        return t.sort_values(["_s", "_m"]).drop(columns=["_s", "_m"]).set_index(keys)

    out += ["## Outcome distribution (share of requests)", "",
            md_table(sort(pd.crosstab([d["size"], d["method"]], d["outcome_group"], normalize="index"))), ""]
    t = d.groupby(keys).agg(requests=("elapsed_s", "size"), elapsed_median=("elapsed_s", "median"),
                            elapsed_p90=("elapsed_s", _q(0.9)), elapsed_max=("elapsed_s", "max"),
                            cpu_median=("cpu_s", "median"), startup_median=("startup_s", "median"))
    out += ["## Elapsed (wall-clock) time per request, seconds", "",
            "`elapsed` is the real time of the request inside its process (step 1 to commit); "
            "`startup` (process start and imports) is excluded from it and listed separately.", "",
            md_table(sort(t), "{:.2f}"), ""]
    t = d.groupby(keys).agg(peak_commit_median=("peak_commit_mb", "median"),
                            peak_commit_max=("peak_commit_mb", "max"),
                            request_commit_median=("request_commit_mb", "median"),
                            peak_wset_median=("peak_wset_mb", "median"),
                            peak_wset_max=("peak_wset_mb", "max"),
                            peak_virtual_median=("peak_virtual_mb", "median"))
    out += ["## Peak memory per request, MB", "",
            "`peak_commit` = peak committed private bytes (virtual memory actually backed, Task Manager "
            "'Commit size'), `request_commit` = its growth during the request, `peak_wset` = peak "
            "physical RAM, `peak_virtual` = peak reserved address space.", "",
            md_table(sort(t), "{:.0f}"), ""]
    cols = ["P_k", "P_0", "dG_plus", "Q_O_delta", "dV_pref", "dV_load", "dV_room", "changed",
            "hours_moved", "day_changes", "room_only_moves", "lecturers_affected", "groups_affected",
            "protected_violations", "lock_violations", "gap_protection_violations", "hard_violations"]
    out += ["## Solution quality (solved requests, mean)", "",
            md_table(sort(solved.groupby(keys)[[c for c in cols if c in solved]].mean()), "{:.2f}"), ""]
    r = d[d["pk_ratio"].notna()]
    if len(r):
        t = r.groupby(keys).agg(n=("pk_ratio", "size"), ratio_median=("pk_ratio", "median"),
                                ratio_max=("pk_ratio", "max"), excess_mean=("pk_excess", "mean"),
                                optimal_share=("pk_ratio", lambda s: float((s <= 1 + 1e-9).mean())))
        out += ["## Distance from the optimum (single-request scenarios)", "",
                "P_k* = proven minimum of P_k over the regulation-feasible set: B4's P*, "
                "used only when B4 proved it optimal (1-hour safety cap). "
                "Methods that ignore regulations (B2, B3) can go below 1.", "",
                md_table(sort(t), "{:.3f}"), ""]
    t = solved.groupby(keys).agg(pstar_proven=("pstar_proven", "mean"),
                                 lexmin_proven=("lexmin_proven", "mean"),
                                 opt_gap_median=("opt_gap_max", "median"),
                                 extra_accepted=("extra_expansion", lambda s: float((s == "accepted").mean())))
    out += ["## Certification of the solved requests (share proven optimal)", "",
            md_table(sort(t), "{:.2f}"), ""]
    reg = d[d["outcome"] == "regulatory_infeasibility"]
    if len(reg):
        t = reg.groupby(keys).agg(n=("mus_size", "size"), mus_size=("mus_size", "mean"),
                                  mus_minimal=("mus_minimal", "mean"), mcs_count=("mcs_count", "mean"),
                                  mcs_verified=("mcs_verified", "sum"), mcs_total=("mcs_count", "sum"))
        out += ["## Explanation quality (regulatory infeasibilities)", "", md_table(sort(t), "{:.2f}"), ""]
    eff = ["iterations", "solver_calls", "solver_seconds", "max_model_vars", "max_model_constraints",
           "F_ratio", "t_search", "t_optimise", "t_extra", "t_explain"]
    out += ["## Effort (mean per request)", "",
            md_table(sort(d.groupby(keys)[[c for c in eff if c in d]].mean()), "{:.2f}"), ""]
    sc = d.groupby(["scenario", "method"]).agg(solved=("outcome", lambda s: float((s == "solved").mean())),
                                               elapsed_median=("elapsed_s", "median"),
                                               peak_commit_median=("peak_commit_mb", "median"))
    sc = sc.reset_index()
    sc["_m"] = sc["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    sc = sc.sort_values(["scenario", "_m"]).drop(columns="_m").set_index(["scenario", "method"])
    out += ["## Per scenario (all sizes)", "", md_table(sc, "{:.2f}"), ""]
    ref = df[df["method"] == "REF"]
    if len(ref):
        t = ref.groupby(["size", "scenario"]).agg(outcome=("outcome", "first"), pstar=("pstar", "first"),
                                                 proven=("pstar_proven", "first"),
                                                 elapsed=("elapsed_s", "median"))
        out += ["## Reference runs (B4, longer limit; older result files only)", "",
                md_table(t, "{:.2f}"), ""]
    return out


def build_report(csv_path: Path, out_dir: Path, lang: str = "hy") -> Path:
    """Tables (English) + four charts; ``lang`` = "hy" (Armenian, default) or "en"
    sets the language of the chart texts."""
    global LANG
    if lang not in TEXTS:
        raise ValueError(f"lang must be one of {sorted(TEXTS)}")
    LANG = lang
    import matplotlib
    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(Path(csv_path))
    global SIZE_ORDER
    SIZE_ORDER = size_order(df)
    # one reference line at the largest per-request search limit used (with a CI
    # time budget the limit differs from request to request)
    used = df.loc[df["method"] != "REF", "time_limit"].dropna().astype(float)
    limits = [float(used.max())] if len(used) else [3600.0]
    CHART_PROBLEMS.clear()
    chart_elapsed(df, out_dir / "elapsed_vs_size.png", limits=tuple(limits))
    chart_memory(df, out_dir / "memory_vs_size.png")
    chart_pk_ratio(df, out_dir / "pk_ratio.png")
    outcome_charts = [f"outcomes_{w}.png" for w in ("synthetic", "itc")
                      if chart_outcomes(df, out_dir / f"outcomes_{w}.png", w)]
    for name, probs in CHART_PROBLEMS.items():
        print(f"chart text check failed for {name}:\n  " + "\n  ".join(probs))
    sizes = _ordered(df["size"].unique(), SIZE_ORDER)
    inst = df.groupby("size")["lessons"].max().reindex(sizes)
    n_ref = int((df["method"] == "REF").sum())
    lines = ["# Heavy-workload performance report", "",
             f"{len(df)} requests" + (f" ({n_ref} legacy reference runs)" if n_ref else "") + ", sizes "
             + ", ".join(f"{s} ({int(n)} lessons)" for s, n in inst.items())
             + f", scenarios {', '.join(sorted(df['scenario'].unique()))}.", "",
             "Every request ran in a fresh process. B6 is an approximation of Akkan et al. [4]. "
             "Chart values are medians per request.", "",
             "![Elapsed time vs size](elapsed_vs_size.png)", "",
             "![Peak memory vs size](memory_vs_size.png)", "",
             "![P_k / P_k*](pk_ratio.png)", "",
             ] + [f"![Outcome mix ({f[9:-4]})]({f})\n" for f in outcome_charts] + [
             "Outcome charts split solved requests into those whose every objective stage "
             "was proven optimal and those stopped by the time limit with a solution; "
             "\"time limit\" means no answer in time.", ""]
    lines += tables(df)
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--out", default="experiments/results/perf_report")
    ap.add_argument("--lang", choices=sorted(TEXTS), default="hy",
                    help="language of the chart texts (default: Armenian)")
    a = ap.parse_args(argv)
    csv_path = Path(a.csv)
    if not csv_path.exists():
        results = csv_path.parent
        found = sorted(p.name for p in results.glob("*.csv")) if results.is_dir() else []
        raise SystemExit(
            f"No results file at {csv_path}.\n"
            "Charts are drawn from measured results; produce them first with one of:\n"
            "  .venv\\Scripts\\python.exe -m pytest -m perf          (1 h to several h, writes experiments\\results\\perf_tests.csv)\n"
            "  .venv\\Scripts\\python.exe -m experiments.perf        (heavy run, hours to days, writes experiments\\results\\perf.csv)\n"
            + (f"CSV files already in {results}: {', '.join(found)}" if found else ""))
    print(build_report(csv_path, Path(a.out), a.lang))


if __name__ == "__main__":
    main()
