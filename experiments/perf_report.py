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
    """P_k* from REF rows (B4 with a long limit, proven optimal P*)."""
    ref = df[(df["method"] == "REF") & (df["outcome"] == "solved") & (df["pstar_proven"] == True)]  # noqa: E712
    ref = ref[["size", "seed", "scenario", "pstar"]].rename(columns={"pstar": "pk_star"})
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


# --------------------------------------------------------------------- charts
def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _legend(ax, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, **kw)
    return leg


def chart_elapsed(df: pd.DataFrame, path: Path, limits=(10, 30, 60)) -> None:
    import matplotlib.pyplot as plt
    d = df[df["method"].isin(METHOD_ORDER)]
    sizes = _ordered(d["size"].unique(), SIZE_ORDER)
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150, facecolor=SURFACE)
    _style(ax, "Elapsed (wall-clock) time per request, median by size", "seconds (log scale)")
    x = np.arange(len(sizes))
    for m in _ordered(d["method"].unique(), METHOD_ORDER):
        med = d[d["method"] == m].groupby("size")["elapsed_s"].median().reindex(sizes)
        ax.plot(x, med.values, color=METHOD_COLOR.get(m, MUTED), linewidth=2, marker=MARKERS.get(m, "o"),
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, label=METHOD_LABEL.get(m, m))
    from matplotlib.ticker import FuncFormatter, LogLocator
    for lim in limits:
        ax.axhline(lim, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(f"{lim:g} s limit", xy=(1.0, lim), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x, [f"{s}\n({int(d[d['size'] == s]['lessons'].max())} lessons)" for s in sizes])
    ax.set_xlim(-0.4, len(sizes) - 0.6)
    _legend(ax, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_memory(df: pd.DataFrame, path: Path, guard_mb: float = 3072) -> None:
    import matplotlib.pyplot as plt
    d = df[df["method"].isin(METHOD_ORDER)]
    sizes = _ordered(d["size"].unique(), SIZE_ORDER)
    x = np.arange(len(sizes))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), dpi=150, facecolor=SURFACE, sharey=False)
    panels = [("peak_commit_mb", "Peak committed memory (virtual, private bytes)"),
              ("peak_wset_mb", "Peak working set (physical RAM)")]
    for ax, (col, title) in zip(axes, panels):
        _style(ax, title, "MB, median per request")
        for m in _ordered(d["method"].unique(), METHOD_ORDER):
            med = d[d["method"] == m].groupby("size")[col].median().reindex(sizes)
            ax.plot(x, med.values, color=METHOD_COLOR.get(m, MUTED), linewidth=2,
                    marker=MARKERS.get(m, "o"), markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=METHOD_LABEL.get(m, m))
        ax.set_xticks(x, sizes)
        ax.set_xlim(-0.4, len(sizes) - 0.6)
    top = max(guard_mb, float(np.nanmax(d["peak_commit_mb"].values)) if len(d) else 0) * 1.1
    axes[0].set_ylim(0, top)
    axes[1].set_ylim(bottom=0)
    axes[0].axhline(guard_mb, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    axes[0].annotate(f"memory guard {guard_mb:g} MB", xy=(0.0, guard_mb), xycoords=("axes fraction", "data"),
                     xytext=(4, 3), textcoords="offset points", va="bottom", fontsize=8, color=MUTED)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, len(labels))), frameon=False,
               fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_pk_ratio(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    d = df[df["method"].isin(METHOD_ORDER) & df["pk_ratio"].notna()]
    methods = _ordered(d["method"].unique(), METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    _style(ax, "Changes relative to the optimum: P_k / P_k* (single-request scenarios, solved)",
           "P_k / P_k*   (1 = minimal number of changes)")
    data = [d[d["method"] == m]["pk_ratio"].values for m in methods]
    if not data:
        ax.text(0.5, 0.5, "no solved single-request case with a proven reference yet",
                transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)
        ax.set_xticks([])
    else:
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.6),
                        whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS),
                        flierprops=dict(marker="o", markersize=4, markerfacecolor=MUTED,
                                        markeredgecolor=SURFACE))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLOR.get(m, MUTED))
            patch.set_alpha(0.35)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(2)
        rng = np.random.default_rng(0)
        for i, (m, vals) in enumerate(zip(methods, data), 1):   # every case as a dot
            jitter = rng.uniform(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
            ax.scatter(i + jitter, vals, s=42, color=METHOD_COLOR.get(m, MUTED), edgecolor=SURFACE,
                       linewidth=1.2, zorder=3)
        ax.set_xticks(range(1, len(methods) + 1),
                      [METHOD_LABEL.get(m, m).replace(" ", "\n", 1) + f"\nn = {len(v)}"
                       for m, v in zip(methods, data)])
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_outcomes(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    d = df[df["method"].isin(METHOD_ORDER)]
    sizes = _ordered(d["size"].unique(), SIZE_ORDER)
    groups = [g for g, _ in OUTCOME_GROUPS]
    colors = dict(zip(groups, SLOTS))
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.1 * len(sizes) + 1.5, 4.6), dpi=150,
                             facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        sub = d[d["size"] == size]
        methods = _ordered(sub["method"].unique(), METHOD_ORDER)
        share = pd.crosstab(sub["method"], sub["outcome_group"], normalize="index").reindex(methods)
        _style(ax, f"{size}", "share of requests" if ax is axes[0][0] else "")
        bottom = np.zeros(len(methods))
        for g in groups:
            if g not in share:
                continue
            vals = share[g].fillna(0).values
            ax.bar(range(len(methods)), vals, bottom=bottom, color=colors[g], width=0.7,
                   edgecolor=SURFACE, linewidth=1.5, label=g)
            bottom += vals
        ax.set_xticks(range(len(methods)), methods)
        ax.set_ylim(0, 1.0)
    handles, labels = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="center right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Outcome mix per size and method", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1 - 1.4 / (3.1 * len(sizes) + 1.5), 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


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
                "P_k* = proven minimum of P_k over the regulation-feasible set (reference run REF: "
                "B4 with a long limit - 300 s in the heavy run, 120 s in the perf tests). "
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
        out += ["## Reference runs (B4, long limit)", "", md_table(t, "{:.2f}"), ""]
    return out


def build_report(csv_path: Path, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(Path(csv_path))
    limits = sorted(set(df.loc[df["method"] != "REF", "time_limit"].dropna().astype(float))) or [30]
    chart_elapsed(df, out_dir / "elapsed_vs_size.png", limits=tuple(sorted(set(limits) | {10, 30, 60})))
    chart_memory(df, out_dir / "memory_vs_size.png")
    chart_pk_ratio(df, out_dir / "pk_ratio.png")
    chart_outcomes(df, out_dir / "outcomes.png")
    sizes = _ordered(df["size"].unique(), SIZE_ORDER)
    inst = df.groupby("size")["lessons"].max().reindex(sizes)
    lines = ["# Heavy-workload performance report", "",
             f"{len(df)} requests ({(df['method'] == 'REF').sum()} reference), sizes "
             + ", ".join(f"{s} ({int(n)} lessons)" for s, n in inst.items())
             + f", scenarios {', '.join(sorted(df['scenario'].unique()))}.", "",
             "Every request ran in a fresh process. B6 is an approximation of Akkan et al. [4].", "",
             "![Elapsed time vs size](elapsed_vs_size.png)", "",
             "![Peak memory vs size](memory_vs_size.png)", "",
             "![P_k / P_k*](pk_ratio.png)", "",
             "![Outcome mix](outcomes.png)", ""]
    lines += tables(df)
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--out", default="experiments/results/perf_report")
    a = ap.parse_args(argv)
    csv_path = Path(a.csv)
    if not csv_path.exists():
        results = csv_path.parent
        found = sorted(p.name for p in results.glob("*.csv")) if results.is_dir() else []
        raise SystemExit(
            f"No results file at {csv_path}.\n"
            "Charts are drawn from measured results; produce them first with one of:\n"
            "  .venv\\Scripts\\python.exe -m pytest -m perf          (~25 min, writes experiments\\results\\perf_tests.csv)\n"
            "  .venv\\Scripts\\python.exe -m experiments.perf        (heavy run, ~9-10 h, writes experiments\\results\\perf.csv)\n"
            + (f"CSV files already in {results}: {', '.join(found)}" if found else ""))
    print(build_report(csv_path, Path(a.out)))


if __name__ == "__main__":
    main()
