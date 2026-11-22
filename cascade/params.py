"""Tunable parameters of the procedure (all defaults agreed in the design review)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class Params:
    # perturbation weights of (3): w^T >= w^R >= 0 (integers for CP-SAT)
    w_time: int = 2
    w_room: int = 1
    # budget ladder B = {b_1 < ... < b_|B|}: doubling from first_budget_factor*w^T,
    # always ending at sum_e w_e^T (required by Claim 2)
    first_budget_factor: int = 2
    # Top_m in (12)
    top_m: int = 3
    # number of highest-ranked neighbours whose fixing is posted as an assumption
    # literal, so that the infeasibility core C_j can name them
    core_pool: int = 30
    # when the (non-minimal) CP-SAT core contains the budget literal, re-check
    # without it before deciding "raise B_j" vs "expand F_j"
    minimize_core: bool = True
    # edge weights of (11)
    alpha_h: float = 1.0
    alpha_alt: float = 0.5
    alpha_p: float = 0.5
    # threshold slack delta_k: 0 for voluntary requests, this for external disruptions
    delta_external: int = 2
    # quality triggers for the one extra expansion
    theta_g: float = 0.0
    theta_q_per_soft: float = 0.05      # theta_Q = this * |S^O|
    # who is protected against new idle gaps by (9): groups | lecturers | both | none
    gap_protect: str = "groups"
    # maximum size mu of a minimal correction set, and how many to enumerate
    mcs_max_size: int = 3
    mcs_max_count: int = 3
    # time limits (seconds).  The exact methods get a 1-hour *safety cap*: in
    # practice CP-SAT either finds a solution or proves infeasibility well before
    # it; a request that still hits the cap is reported as an algorithmic failure.
    time_limit: float = 3600.0          # shared by all sub-problems of one request
    explain_time_limit: float = 3600.0  # MUS / MCS / physical check
    b6_time_limit: float = 30.0         # B6 is a heuristic: it can never prove infeasibility
    workers: int = 8                    # CP-SAT search workers
    light_presolve_above: int = 5000    # models with more x-variables skip probing/symmetry
    single_worker_below: int = 3000     # models with fewer x-variables use one worker
    # integer scale for Q_O = sum pi_s * num_s / den_s
    q_scale: int = 1000
    # optimistic concurrency retries (step 6)
    max_retries: int = 3
    # B6 heuristic: perturbation limit = w^T * (|F0| + b6_extra_moves) + delta_k
    b6_extra_moves: int = 3
    seed: int = 0
    lang: str = "en"

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)
