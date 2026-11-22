"""Analysis of runner CSVs: outcome distribution, metrics, time split by
feasible / infeasible requests, fairness between lecturers and groups, and
paired Wilcoxon tests (B7 vs every baseline) with Holm correction.

    python -m experiments.analysis experiments/results/run.csv --out experiments/results/report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROPOSED = "B7"
SEQ = ["size", "seed", "time_limit", "U_P"]
# per-sequence summaries compared pairwise; "lower" says which direction is better
TESTED = {
    "solve_rate": "higher",
    "P_k": "lower",
    "dG_plus": "lower",
    "Q_O_delta": "lower",
    "F_ratio": "lower",
    "seconds": "lower",
    "protected_violations": "lower",
}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted((p, k) for k, p in pvals.items() if not np.isnan(p))
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k, p in pvals.items():
        out.setdefault(k, np.nan)
    return out


def per_sequence(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["solved"] = d["outcome"] == "solved"
    d["Q_O_delta"] = d["Q_O"] - d["Q_O_k"]
    solved = d[d["solved"]]
    agg = solved.groupby(SEQ + ["method"])[["P_k", "dG_plus", "Q_O_delta", "F_ratio",
                                             "protected_violations"]].mean()
    agg["solve_rate"] = d.groupby(SEQ + ["method"])["solved"].mean()
    agg["seconds"] = d.groupby(SEQ + ["method"])["seconds"].mean()
    return agg.reset_index()


def wilcoxon_table(seq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = sorted(m for m in seq["method"].unique() if m != PROPOSED)
    for metric, better in TESTED.items():
        pvals, stats = {}, {}
        for b in baselines:
            a = seq[seq["method"] == PROPOSED].set_index(SEQ)[metric]
            c = seq[seq["method"] == b].set_index(SEQ)[metric]
            pair = pd.concat([a, c], axis=1, keys=["B7", b]).dropna()
            diff = pair["B7"] - pair[b]
            n = len(pair)
            if n < 2 or np.allclose(diff, 0):
                p = np.nan if n < 2 else 1.0
            else:
                try:
                    p = wilcoxon(pair["B7"], pair[b], zero_method="zsplit").pvalue
                except ValueError:
                    p = np.nan
            pvals[b] = p
            stats[b] = (n, pair["B7"].median() if n else np.nan, pair[b].median() if n else np.nan)
        adj = holm(pvals)
        for b in baselines:
            n, m7, mb = stats[b]
            if np.isnan(m7) or np.isnan(mb):
                verdict = ""
            elif adj[b] < 0.05:
                good = m7 > mb if better == "higher" else m7 < mb
                verdict = "B7 better" if good else "B7 worse"
            else:
                verdict = "n.s."
            rows.append({"metric": metric, "baseline": b, "pairs": n, "median_B7": m7,
                         "median_baseline": mb, "p": pvals[b], "p_holm": adj[b], "verdict": verdict})
    return pd.DataFrame(rows).set_index(["metric", "baseline"])


def report(df: pd.DataFrame) -> str:
    out = ["# Experiment report", "",
           f"{len(df)} requests, {df.groupby(SEQ + ['method']).ngroups} sequences, "
           f"methods: {', '.join(sorted(df['method'].unique()))}.", "",
           "B6 is an approximation of Akkan et al. [4] (simulated annealing with a perturbation "
           "limit), not their exact algorithm.", ""]
    for keys, g in df.groupby(["size", "time_limit", "U_P"]):
        size, tl, up = keys
        out += [f"## Size {size}, time limit {tl:g} s, U^P = {up}", "", "### Outcome distribution (share of requests)", ""]
        dist = pd.crosstab(g["method"], g["outcome"], normalize="index")
        out += [md_table(dist), ""]
        solved = g[g["outcome"] == "solved"].copy()
        solved["Q_O_delta"] = solved["Q_O"] - solved["Q_O_k"]
        cols = ["P_k", "P_0", "G", "dG_plus", "Q_O_delta", "F_ratio", "changed",
                "protected_violations", "lock_violations", "gap_protection_violations"]
        out += ["### Solved requests: mean metrics", "", md_table(solved.groupby("method")[cols].mean()), ""]
        g = g.copy()
        g["class"] = np.where(g["outcome"] == "solved", "solved", "not solved")
        t = g.pivot_table(index="method", columns="class", values="seconds", aggfunc="median")
        out += ["### Median time (s): solved vs not solved requests", "", md_table(t), ""]
        fair = solved.groupby("method")[["lecturers_changed", "groups_changed",
                                         "dG_lecturers", "dG_groups"]].mean()
        out += ["### Distribution of changes and new gaps (mean per solved request)", "", md_table(fair), ""]
        trig = g[g["method"] == PROPOSED]["extra_expansion"].value_counts()
        if len(trig):
            out += ["### B7 extra expansion", "", md_table(trig.to_frame("requests")), ""]
    seq = per_sequence(df)
    if PROPOSED in seq["method"].unique() and seq["method"].nunique() > 1:
        out += ["## Paired Wilcoxon tests on per-sequence means (B7 vs baseline, Holm-corrected)", "",
                md_table(wilcoxon_table(seq), "{:.4g}"), ""]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    df = pd.concat([pd.read_csv(p) for p in a.csv], ignore_index=True)
    text = report(df)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"report written to {a.out}")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
