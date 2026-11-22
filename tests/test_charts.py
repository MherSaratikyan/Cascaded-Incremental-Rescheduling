"""Chart layout: no text may overlap another text, a legend may not cover a
panel, and nothing may stick out of the figure - checked on worst-case data
(all 7 methods, every outcome, many sizes with long names) in Armenian and
English.  The check measures the drawn bounding boxes (perf_report.text_overlaps)."""
import numpy as np
import pandas as pd
import pytest

from experiments import perf_report as pr

SIZES = {"XS": 20, "S": 60, "M": 200, "L": 500, "XL": 1000, "itc-synth-s": 216,
         "itc-synth-m": 475, "itc-synth-l": 1240, "itc-muni-fspsx-fal17": 1623}
OUTCOMES = ["solved", "solved", "physical_infeasibility", "regulatory_infeasibility", "lock_conflict",
            "rejected_local_check", "algorithmic_failure", "memory_limit", "error"]


def worst_case_rows() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for size, lessons in SIZES.items():
        for m in pr.METHOD_ORDER:
            for k, outcome in enumerate(OUTCOMES):
                rows.append(dict(
                    size=size, method=m, scenario="H1", seed=1, step=0, lessons=lessons,
                    outcome=outcome, lexmin_proven=bool(k % 2),
                    elapsed_s=float(rng.uniform(0.05, 3500)), time_limit=3600.0,
                    peak_commit_mb=float(rng.uniform(250, 2900)), peak_wset_mb=float(rng.uniform(90, 2400)),
                    pk_ratio=float(1 + rng.uniform(0, 0.6)) if outcome == "solved" else np.nan))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("lang", ["hy", "en"])
def test_charts_have_no_overlapping_text(tmp_path, lang):
    """All four charts (elapsed, memory, P_k/P_k*, outcome mix for synthetic and
    ITC instances) are drawn without any overlapping labels, axis titles,
    legend entries or panel titles, and without text outside the figure."""
    import matplotlib
    matplotlib.use("Agg")
    df = worst_case_rows()
    pr.LANG = lang
    pr.SIZE_ORDER = pr.size_order(df)
    pr.CHART_PROBLEMS.clear()
    pr.chart_elapsed(df, tmp_path / "elapsed.png", limits=(3600.0,))
    pr.chart_memory(df, tmp_path / "memory.png")
    pr.chart_pk_ratio(df, tmp_path / "pk.png")
    assert pr.chart_outcomes(df, tmp_path / "out_syn.png", "synthetic")
    assert pr.chart_outcomes(df, tmp_path / "out_itc.png", "itc")
    problems = {k: v for k, v in pr.CHART_PROBLEMS.items()}
    assert not problems, "\n".join(f"{k}: {p}" for k, ps in problems.items() for p in ps)
    for f in ("elapsed.png", "memory.png", "pk.png", "out_syn.png", "out_itc.png"):
        assert (tmp_path / f).stat().st_size > 10_000
