"""HBCU Scholarship Economics & Student Success Lab — Single-File Edition.

Run locally with: streamlit run Econ_keystone.py

Architecture:
- Part 1 (MODEL): Transparent, deterministic, non-causal policy stress-test engine,
  including an optional Poverty & Capitalization Constraints module (weekly work-hour
  cap, capital-access-scaled loan cap, and absorbable-gap retention penalty kink).
- Part 2 (NARRATIVE & CHARTS): LaTeX-safe dynamic Markdown explanations (escaping
  currency '$' as '\\$' so Streamlit never mangles prose into math mode) and
  clean, non-overlapping Plotly visualizations.
- Part 3 (INTERFACE): Structured Streamlit layout using native bordered containers.

Governance note: All outputs are scenario hypotheses conditional on user inputs,
not causal estimates of scholarship policy at any specific institution.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ===========================================================================
# PART 1 · DETERMINISTIC ECONOMIC MODEL
# ===========================================================================

REQUIRED_COLUMNS = (
    "hs_gpa",
    "college_gpa",
    "scholarship_amount",
    "dependent",
    "need_index",
)
OPTIONAL_COLUMNS = (
    "readiness_score",
    "section_group",
    "reenrolled_next_term",
    "renewed_actual",
    "cohort_year",
    "pell_eligible",
    "capital_constrained",
    "maxed_federal_loans",
    "plus_denied",
)
ACCEPTED_COLUMNS = set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


@dataclass(frozen=True)
class CohortSettings:
    n: int = 650
    seed: int = 42
    entry_hs_cutoff: float = 3.50
    hs_grade_uplift: float = 0.20
    hs_school_spread: float = 0.13
    college_section_spread: float = 0.14
    college_grade_uplift: float = 0.00
    scholarship_amount: float = 8_000.0
    dependent_share: float = 0.70
    poverty_share: float = 0.55
    capital_constrained_share: float = 0.35


@dataclass(frozen=True)
class Economics:
    tuition_per_retained: float = 15_000.0
    marginal_instruction_cost: float = 6_500.0
    baseline_retention: float = 0.82
    full_loss_retention_penalty: float = 0.25
    work_replacement_share: float = 0.30
    loan_replacement_share: float = 0.45
    hourly_wage: float = 15.0
    academic_work_weeks: int = 32
    earnings_premium_per_year: float = 12_000.0
    earnings_horizon_years: int = 10
    real_discount_rate: float = 0.035


@dataclass(frozen=True)
class Policy:
    threshold: float = 3.50
    probation_band: float = 0.00
    probation_fraction: float = 0.00
    support_band: float = 0.00
    support_gpa_lift: float = 0.00
    support_cost_per_student: float = 0.00


STRICT_POLICY = Policy(threshold=3.50)


@dataclass(frozen=True)
class FinancingConstraints:
    """Optional poverty (flow) and capitalization (stock/credit) limits on aid replacement.

    Disabled (default) reproduces the unconstrained baseline model identically.
    """

    enabled: bool = False
    constrained_cap_access: float = 0.25
    work_hours_cap: float = 15.0
    annual_loan_cap: float = 5_500.0
    absorbable_gap_base: float = 3_000.0
    absorbable_gap_poverty: float = 750.0
    gap_retention_penalty: float = 0.10
    impute_poverty_share: float = 0.55
    impute_constrained_share: float = 0.35


NO_CONSTRAINTS = FinancingConstraints()


def validate_constraints(fc: FinancingConstraints) -> None:
    if not (0.0 <= fc.constrained_cap_access <= 1.0):
        raise ValueError("Capital access must be between 0.0 and 1.0.")
    if fc.work_hours_cap <= 0 or fc.annual_loan_cap < 0:
        raise ValueError("Work-hour cap must be positive and annual loan cap nonnegative.")
    if min(fc.absorbable_gap_base, fc.absorbable_gap_poverty) < 0:
        raise ValueError("Absorbable gap thresholds must be nonnegative.")
    if not (0.0 <= fc.gap_retention_penalty <= 0.5):
        raise ValueError("Gap retention penalty must be between 0.0 and 0.50.")
    for share in (fc.impute_poverty_share, fc.impute_constrained_share):
        if not (0.0 <= share <= 1.0):
            raise ValueError("Imputation shares must be between 0.0 and 1.0.")


def _top_share_flags(latent: np.ndarray, share: float) -> np.ndarray:
    """Deterministically flag (as 1) the top `share` of a continuous latent vector."""
    n = len(latent)
    k = int(round(share * n))
    if k <= 0:
        return np.zeros(n, dtype=int)
    if k >= n:
        return np.ones(n, dtype=int)
    threshold = np.partition(latent, n - k)[n - k]
    return (latent >= threshold).astype(int)


def derive_financing_flags(
    data: pd.DataFrame, fc: FinancingConstraints
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Resolve poverty-proxy and capital-constrained flags for a cohort.

    Uses supplied 0/1 columns (`pell_eligible`, `capital_constrained`, or revealed
    constraint markers `maxed_federal_loans` / `plus_denied`) when available;
    otherwise imputes deterministically by `need_index` rank.
    """
    need = data["need_index"].to_numpy(float)
    if "pell_eligible" in data.columns:
        poverty = (data["pell_eligible"].to_numpy(int) == 1).astype(int)
        poverty_imputed = False
    else:
        poverty = _top_share_flags(need, fc.impute_poverty_share)
        poverty_imputed = True

    revealed = [c for c in ("maxed_federal_loans", "plus_denied") if c in data.columns]
    if "capital_constrained" in data.columns:
        constrained = (data["capital_constrained"].to_numpy(int) == 1).astype(int)
        constrained_imputed = False
    elif revealed:
        stacked = np.column_stack([data[c].to_numpy(int) for c in revealed])
        constrained = (stacked.max(axis=1) == 1).astype(int)
        constrained_imputed = False
    else:
        constrained = _top_share_flags(need, fc.impute_constrained_share)
        constrained_imputed = True

    return poverty, constrained, poverty_imputed, constrained_imputed


def build_synthetic_cohort(c: CohortSettings) -> pd.DataFrame:
    """Create a fixed-size synthetic cohort of entering scholarship recipients."""
    if not (50 <= c.n <= 15_000):
        raise ValueError("Cohort size must be between 50 and 15,000.")
    if not (0.0 <= c.dependent_share <= 1.0):
        raise ValueError("Dependent share must be between 0.0 and 1.0.")
    if not (2.5 <= c.entry_hs_cutoff <= 3.90):
        raise ValueError("Entry high-school GPA cutoff must be between 2.50 and 3.90.")
    for share in (c.poverty_share, c.capital_constrained_share):
        if not (0.0 <= share <= 1.0):
            raise ValueError("Poverty and capital-constrained shares must be between 0.0 and 1.0.")

    rng = np.random.default_rng(c.seed)
    pool = max(10_000, 30 * c.n)
    readiness = rng.beta(4.3, 2.5, pool)
    school_id = rng.integers(0, 18, pool)
    school_shifts = rng.normal(0.0, c.hs_school_spread, 18)

    hs_gpa = np.clip(
        2.22
        + 1.62 * readiness
        + c.hs_grade_uplift
        + school_shifts[school_id]
        + rng.normal(0.0, 0.18, pool),
        0.0,
        4.0,
    )
    selected = np.flatnonzero(hs_gpa >= c.entry_hs_cutoff)
    if len(selected) < c.n:
        raise ValueError(
            "Too few eligible applicants in synthetic pool: lower the entry HS GPA cutoff or cohort size."
        )

    idx = selected[: c.n]
    selected_readiness = readiness[idx]
    section = rng.integers(0, 18, c.n)
    section_shifts = rng.normal(0.0, c.college_section_spread, 18)

    mastery = np.clip(
        1.63 + 2.22 * selected_readiness + rng.normal(0.0, 0.21, c.n),
        0.0,
        4.0,
    )
    observed_college = np.clip(
        mastery
        + c.college_grade_uplift
        + section_shifts[section]
        + rng.normal(0.0, 0.10, c.n),
        0.0,
        4.0,
    )
    dependent = (rng.random(c.n) < c.dependent_share).astype(int)
    need = rng.beta(3.1, 2.1, c.n)

    # Poverty proxy (flow constraint) tracks need closely; capital access (stock/credit)
    # is a distinct construct only partially correlated with income/need.
    poverty_latent = need + rng.normal(0.0, 0.15, c.n)
    pell = _top_share_flags(poverty_latent, c.poverty_share)
    need_z = (need - need.mean()) / max(float(need.std()), 1e-9)
    capital_latent = 0.6 * need_z + 0.8 * rng.normal(0.0, 1.0, c.n)
    constrained = _top_share_flags(capital_latent, c.capital_constrained_share)

    return pd.DataFrame(
        {
            "hs_gpa": hs_gpa[idx].round(3),
            "college_gpa": observed_college.round(3),
            "scholarship_amount": np.full(c.n, c.scholarship_amount),
            "dependent": dependent,
            "need_index": need.round(4),
            "pell_eligible": pell,
            "capital_constrained": constrained,
            "readiness_score": (100.0 * selected_readiness).round(2),
            "section_group": [f"SEC-{i + 1:02d}" for i in section],
            "cohort_year": np.full(c.n, 2026),
            "mastery_gpa_proxy": mastery.round(3),
        }
    )


def _binary(series: pd.Series, column: str) -> pd.Series:
    mapping = {
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        raise ValueError(f"Column '{column}' must contain only 0/1, yes/no, or true/false.")
    return normalized.astype(int)


def validate_uploaded_cohort(data: pd.DataFrame) -> pd.DataFrame:
    """Validate uploaded CSV schema and reject unexpected/identifying columns."""
    if len(data) < 25:
        raise ValueError("At least 25 de-identified records are required.")
    if len(data) > 50_000:
        raise ValueError("Upload a dataset of at most 50,000 records.")

    extra = sorted(set(data.columns) - ACCEPTED_COLUMNS)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if extra:
        raise ValueError(
            "Unapproved columns detected (remove IDs, names, and extra fields): "
            + ", ".join(extra[:10])
        )
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    frame = data.copy()
    for col in ("hs_gpa", "college_gpa", "scholarship_amount", "need_index"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any() or not np.isfinite(frame[col].to_numpy()).all():
            raise ValueError(f"Column '{col}' must contain finite numbers in every row.")

    for col in ("hs_gpa", "college_gpa"):
        if not frame[col].between(0.0, 4.0).all():
            raise ValueError(f"Column '{col}' must be on a 0.00–4.00 GPA scale.")

    if not frame["scholarship_amount"].between(0.0, 75_000.0).all():
        raise ValueError("Column 'scholarship_amount' must be between 0 and 75,000.")
    if not frame["need_index"].between(0.0, 1.0).all():
        raise ValueError("Column 'need_index' must be between 0.0 and 1.0.")

    frame["dependent"] = _binary(frame["dependent"], "dependent")
    for optional_binary in (
        "reenrolled_next_term",
        "renewed_actual",
        "pell_eligible",
        "capital_constrained",
        "maxed_federal_loans",
        "plus_denied",
    ):
        if optional_binary in frame.columns:
            frame[optional_binary] = _binary(frame[optional_binary], optional_binary)

    if "readiness_score" in frame.columns:
        frame["readiness_score"] = pd.to_numeric(frame["readiness_score"], errors="coerce")
        if frame["readiness_score"].isna().any() or not frame["readiness_score"].between(0.0, 100.0).all():
            raise ValueError("Column 'readiness_score' must be between 0 and 100.")

    if "cohort_year" in frame.columns:
        frame["cohort_year"] = pd.to_numeric(frame["cohort_year"], errors="coerce")
        if frame["cohort_year"].isna().any() or not frame["cohort_year"].between(1990, 2100).all():
            raise ValueError("Column 'cohort_year' must be a four-digit year (1990–2100).")

    if "section_group" in frame.columns:
        groups = frame["section_group"].astype(str)
        if groups.str.len().gt(20).any() or not groups.str.fullmatch(r"[A-Za-z0-9_\- ]+").all():
            raise ValueError("Column 'section_group' must contain short anonymous codes only.")
        frame["section_group"] = groups

    return frame


def validate_economics(e: Economics) -> None:
    if e.tuition_per_retained < 0 or e.marginal_instruction_cost < 0:
        raise ValueError("Tuition and marginal cost must be nonnegative.")
    if not (0.0 <= e.baseline_retention <= 1.0 and 0.0 <= e.full_loss_retention_penalty <= 1.0):
        raise ValueError("Retention parameters must be between 0 and 1.")
    if min(e.work_replacement_share, e.loan_replacement_share) < 0:
        raise ValueError("Work and loan shares must be nonnegative.")
    if e.work_replacement_share + e.loan_replacement_share > 1.0000001:
        raise ValueError("Work + loan replacement shares cannot exceed 100%.")
    if e.hourly_wage <= 0 or e.academic_work_weeks <= 0:
        raise ValueError("Hourly wage and academic work weeks must be positive.")
    if e.earnings_horizon_years <= 0 or e.real_discount_rate < 0:
        raise ValueError("Earnings horizon must be positive and discount rate nonnegative.")


def evaluate_policy(
    data: pd.DataFrame,
    policy: Policy,
    e: Economics,
    fc: Optional[FinancingConstraints] = None,
) -> dict:
    """Compute expected student and institutional outcomes for a single policy."""
    fc = fc or NO_CONSTRAINTS
    validate_economics(e)
    validate_constraints(fc)
    if not (2.0 <= policy.threshold <= 4.0):
        raise ValueError("Threshold must be between 2.00 and 4.00.")
    if not (0.0 <= policy.probation_band <= 1.0 and 0.0 <= policy.probation_fraction <= 1.0):
        raise ValueError("Invalid probation settings.")
    if not (0.0 <= policy.support_band <= 1.0 and 0.0 <= policy.support_gpa_lift <= 0.5):
        raise ValueError("Invalid tutoring support settings.")
    if policy.support_cost_per_student < 0:
        raise ValueError("Support cost cannot be negative.")

    raw_gpa = data["college_gpa"].to_numpy(float)
    need = data["need_index"].to_numpy(float)
    dependent = data["dependent"].to_numpy(bool)
    amount = data["scholarship_amount"].to_numpy(float)
    n = len(data)

    targeted = (
        (policy.support_gpa_lift > 0)
        & (policy.support_band > 0)
        & (raw_gpa < policy.threshold)
        & (raw_gpa >= policy.threshold - policy.support_band)
    )
    effective_gpa = np.clip(raw_gpa + targeted * policy.support_gpa_lift, 0.0, 4.0)
    rescued_to_full = (raw_gpa < policy.threshold - 1e-10) & (
        effective_gpa >= policy.threshold - 1e-10
    )

    full = effective_gpa >= policy.threshold - 1e-10
    probation = (
        (~full)
        & (policy.probation_band > 0)
        & (effective_gpa >= policy.threshold - policy.probation_band - 1e-10)
    )
    award_fraction = np.where(full, 1.0, np.where(probation, policy.probation_fraction, 0.0))
    award = amount * award_fraction
    lost_award = amount - award
    loss_ratio = np.divide(lost_award, amount, out=np.zeros(n), where=amount > 0)

    # Optional poverty & capitalization constraints on replacing lost scholarship dollars
    if fc.enabled:
        poverty, constrained, _, _ = derive_financing_flags(data, fc)
        cap_access = np.where(constrained == 1, fc.constrained_cap_access, 1.0)
        hours_cap: float = fc.work_hours_cap
        loan_cap: float = fc.annual_loan_cap
        absorbable = np.where(
            poverty == 1, fc.absorbable_gap_poverty, fc.absorbable_gap_base
        )
        gap_penalty = fc.gap_retention_penalty
    else:
        poverty = np.zeros(n, dtype=int)
        constrained = np.zeros(n, dtype=int)
        cap_access = np.ones(n)
        hours_cap = np.inf
        loan_cap = np.inf
        absorbable = np.full(n, np.inf)
        gap_penalty = 0.0

    implied_work_hrs_week = (
        lost_award * e.work_replacement_share / e.hourly_wage / e.academic_work_weeks
    )
    potential_work_hrs_week = np.minimum(implied_work_hrs_week, hours_cap)
    work_replacement_dollars = (
        potential_work_hrs_week * e.hourly_wage * e.academic_work_weeks
    )
    potential_new_loan = np.minimum(
        lost_award * e.loan_replacement_share * cap_access, loan_cap
    )
    potential_uncovered = np.maximum(
        lost_award - work_replacement_dollars - potential_new_loan, 0.0
    )
    excess_gap = np.where(
        np.isfinite(absorbable), np.maximum(potential_uncovered - absorbable, 0.0), 0.0
    )
    gap_strain = gap_penalty * np.minimum(
        1.0, np.divide(excess_gap, amount, out=np.zeros(n), where=amount > 0)
    )

    base_retention = np.clip(
        e.baseline_retention + 0.16 * (raw_gpa - 3.0) - 0.08 * (need - 0.5),
        0.10,
        0.98,
    )
    vulnerability_weight = np.where(dependent, 1.0, 0.25) * (0.75 + 0.5 * need)
    retention_prob = np.clip(
        base_retention
        + 0.16 * (effective_gpa - raw_gpa)
        - e.full_loss_retention_penalty * loss_ratio * vulnerability_weight
        - gap_strain,
        0.02,
        0.99,
    )
    completion_given_retention = np.clip(
        0.60 + 0.19 * (effective_gpa - 3.0) - 0.08 * (need - 0.5),
        0.12,
        0.92,
    )
    completion_proxy = retention_prob * completion_given_retention

    support_spend = float(targeted.sum() * policy.support_cost_per_student)
    expected_discounts = float((award * retention_prob).sum())
    expected_gross_tuition = float((retention_prob * e.tuition_per_retained).sum())
    expected_marginal_cost = float((retention_prob * e.marginal_instruction_cost).sum())
    contribution = float(
        (retention_prob * (e.tuition_per_retained - e.marginal_instruction_cost - award)).sum()
        - support_spend
    )

    affected_mask = lost_award > 1e-6
    affected_count = int(affected_mask.sum())
    mean_work_hrs_affected = (
        float(potential_work_hrs_week[affected_mask].mean()) if affected_count > 0 else 0.0
    )
    mean_loan_affected = (
        float(potential_new_loan[affected_mask].mean()) if affected_count > 0 else 0.0
    )

    metrics = {
        "n": n,
        "full_renewal_count": int((award_fraction >= 0.999999).sum()),
        "partial_renewal_count": int(((award_fraction > 0) & (award_fraction < 0.999999)).sum()),
        "full_loss_count": int((award_fraction < 1e-8).sum()),
        "affected_any_loss_count": affected_count,
        "targeted_count": int(targeted.sum()),
        "rescued_to_full_count": int(rescued_to_full.sum()),
        "expected_retained": float(retention_prob.sum()),
        "expected_retention_rate": float(retention_prob.mean()),
        "expected_completion_proxy": float(completion_proxy.sum()),
        "expected_completion_proxy_rate": float(completion_proxy.mean()),
        "expected_institutional_discounts": expected_discounts,
        "support_spend": support_spend,
        "total_institutional_spend": expected_discounts + support_spend,
        "expected_gross_tuition": expected_gross_tuition,
        "expected_marginal_instruction_cost": expected_marginal_cost,
        "expected_net_contribution": contribution,
        "potential_lost_award_total": float(lost_award.sum()),
        "potential_work_replacement_dollars": float(work_replacement_dollars.sum()),
        "potential_extra_work_hours_week_total": float(potential_work_hrs_week.sum()),
        "potential_extra_borrowing_total": float(potential_new_loan.sum()),
        "potential_uncovered_gap_total": float(potential_uncovered.sum()),
        "mean_work_hours_per_affected_student": mean_work_hrs_affected,
        "mean_loan_per_affected_student": mean_loan_affected,
        "expected_extra_work_hours_week_among_reenrollees": float(
            (potential_work_hrs_week * retention_prob).sum()
        ),
        "expected_extra_borrowing_among_reenrollees": float(
            (potential_new_loan * retention_prob).sum()
        ),
        "dependent_full_loss_count": int(((award_fraction < 1e-8) & dependent).sum()),
        "mean_effective_gpa": float(effective_gpa.mean()),
        "poverty_proxy_count": int(poverty.sum()),
        "capital_constrained_count": int(constrained.sum()),
        "excess_gap_total": float(excess_gap.sum()),
        "students_over_absorbable_gap": int((excess_gap > 1e-9).sum()),
    }
    detailed = pd.DataFrame(
        {
            "college_gpa": raw_gpa,
            "effective_gpa": effective_gpa,
            "award_fraction": award_fraction,
            "award_expected_if_retained": award,
            "retention_probability": retention_prob,
            "completion_proxy_probability": completion_proxy,
            "lost_award_if_enrolled": lost_award,
            "potential_added_work_hours_per_week": potential_work_hrs_week,
            "potential_added_loans": potential_new_loan,
            "potential_uncovered_gap": potential_uncovered,
            "excess_unfunded_gap": excess_gap,
            "poverty_proxy": poverty.astype(bool),
            "capital_constrained": constrained.astype(bool),
            "targeted_for_support": targeted,
        }
    )
    return {"metrics": metrics, "detail": detailed, "policy": policy}


def compare_policies(baseline: dict, alternative: dict) -> dict[str, float]:
    b, a = baseline["metrics"], alternative["metrics"]
    return {
        "delta_" + key: a[key] - b[key]
        for key in b
        if isinstance(b[key], (int, float, np.number)) and key != "n"
    }


def ten_year_annuity_factor(e: Economics) -> float:
    r, n = e.real_discount_rate, e.earnings_horizon_years
    return float(n if r == 0 else (1.0 - (1.0 + r) ** (-n)) / r)


def optimize_policies(
    cohort: pd.DataFrame,
    e: Economics,
    baseline: dict,
    probation_band: float,
    support_band: float,
    support_gpa_lift: float,
    support_cost_per_student: float,
    max_added_spend: float,
    fc: Optional[FinancingConstraints] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Grid-search discrete policy combinations under an incremental budget cap."""
    if max_added_spend < 0:
        raise ValueError("Optimization budget cap must be nonnegative.")

    base = baseline["metrics"]
    rows = []
    support_modes = [False] + ([True] if support_gpa_lift > 0 and support_band > 0 else [])

    for threshold in np.round(np.arange(2.50, 3.50001, 0.05), 2):
        for grace_share in (0.0, 0.25, 0.50, 0.75, 1.0):
            for use_support in support_modes:
                policy = Policy(
                    threshold=float(threshold),
                    probation_band=probation_band,
                    probation_fraction=grace_share,
                    support_band=support_band if use_support else 0.0,
                    support_gpa_lift=support_gpa_lift if use_support else 0.0,
                    support_cost_per_student=support_cost_per_student if use_support else 0.0,
                )
                m = evaluate_policy(cohort, policy, e, fc)["metrics"]
                extra = m["total_institutional_spend"] - base["total_institutional_spend"]
                rows.append(
                    {
                        "renewal_gpa": float(threshold),
                        "probation_award_pct": int(100 * grace_share),
                        "pre_renewal_support": bool(use_support),
                        "additional_spend": float(extra),
                        "additional_expected_retained": m["expected_retained"]
                        - base["expected_retained"],
                        "additional_completion_proxy": m["expected_completion_proxy"]
                        - base["expected_completion_proxy"],
                        "additional_net_contribution": m["expected_net_contribution"]
                        - base["expected_net_contribution"],
                        "total_expected_retained": m["expected_retained"],
                        "total_expected_completions_proxy": m["expected_completion_proxy"],
                        "support_students": m["targeted_count"],
                    }
                )

    candidates = pd.DataFrame(rows)
    baseline_row = pd.DataFrame(
        [
            {
                "renewal_gpa": 3.50,
                "probation_award_pct": 0,
                "pre_renewal_support": False,
                "additional_spend": 0.0,
                "additional_expected_retained": 0.0,
                "additional_completion_proxy": 0.0,
                "additional_net_contribution": 0.0,
                "total_expected_retained": base["expected_retained"],
                "total_expected_completions_proxy": base["expected_completion_proxy"],
                "support_students": 0,
            }
        ]
    )
    candidates = pd.concat([baseline_row, candidates], ignore_index=True).drop_duplicates(
        subset=["renewal_gpa", "probation_award_pct", "pre_renewal_support"]
    )
    feasible = candidates.loc[candidates["additional_spend"] <= max_added_spend + 1e-7]
    best = feasible.sort_values(
        ["additional_expected_retained", "additional_completion_proxy", "additional_spend"],
        ascending=[False, False, True],
    ).iloc[0]
    return candidates, best


def threshold_sweep(
    cohort: pd.DataFrame,
    e: Economics,
    probation_band: float,
    probation_fraction: float,
    support_band: float = 0.0,
    support_gpa_lift: float = 0.0,
    support_cost_per_student: float = 0.0,
    thresholds: Optional[np.ndarray] = None,
    fc: Optional[FinancingConstraints] = None,
) -> pd.DataFrame:
    """Evaluate expected outcomes across full-renewal GPA thresholds from 2.50 to 3.80."""
    if thresholds is None:
        thresholds = np.round(np.arange(2.50, 3.8001, 0.05), 2)
    rows = []
    for t in thresholds:
        policy = Policy(
            threshold=float(t),
            probation_band=probation_band,
            probation_fraction=probation_fraction,
            support_band=support_band,
            support_gpa_lift=support_gpa_lift,
            support_cost_per_student=support_cost_per_student,
        )
        m = evaluate_policy(cohort, policy, e, fc)["metrics"]
        rows.append(
            {
                "threshold": float(t),
                "expected_retained": m["expected_retained"],
                "expected_retention_rate": m["expected_retention_rate"],
                "expected_completion_proxy": m["expected_completion_proxy"],
                "total_institutional_spend": m["total_institutional_spend"],
                "expected_net_contribution": m["expected_net_contribution"],
                "full_loss_count": m["full_loss_count"],
            }
        )
    return pd.DataFrame(rows)


def assumption_sensitivity(
    cohort: pd.DataFrame,
    policy: Policy,
    e: Economics,
    fc: Optional[FinancingConstraints] = None,
) -> pd.DataFrame:
    """Perturb one assumption at a time around the central scenario comparison."""
    fc = fc or NO_CONSTRAINTS

    def retained_delta(
        econ: Economics, pol: Policy, f: FinancingConstraints
    ) -> float:
        strict = evaluate_policy(cohort, STRICT_POLICY, econ, f)["metrics"]
        alt = evaluate_policy(cohort, pol, econ, f)["metrics"]
        return alt["expected_retained"] - strict["expected_retained"]

    central = retained_delta(e, policy, fc)
    specs: list[
        tuple[
            str,
            Economics,
            Policy,
            FinancingConstraints,
            Economics,
            Policy,
            FinancingConstraints,
        ]
    ] = [
        (
            "Aid-loss retention penalty (±50%)",
            replace(e, full_loss_retention_penalty=min(1.0, 0.5 * e.full_loss_retention_penalty)),
            policy,
            fc,
            replace(e, full_loss_retention_penalty=min(1.0, 1.5 * e.full_loss_retention_penalty)),
            policy,
            fc,
        ),
        (
            "Baseline retention at GPA 3.0 (±5 pp)",
            replace(e, baseline_retention=max(0.0, e.baseline_retention - 0.05)),
            policy,
            fc,
            replace(e, baseline_retention=min(1.0, e.baseline_retention + 0.05)),
            policy,
            fc,
        ),
    ]
    if policy.support_gpa_lift > 0 and policy.support_band > 0:
        specs.append(
            (
                "Tutoring GPA improvement (±50%)",
                e,
                replace(policy, support_gpa_lift=min(0.5, 0.5 * policy.support_gpa_lift)),
                fc,
                e,
                replace(policy, support_gpa_lift=min(0.5, 1.5 * policy.support_gpa_lift)),
                fc,
            )
        )
    if policy.probation_band > 0:
        specs.append(
            (
                "Probation award fraction (±25 pp)",
                e,
                replace(policy, probation_fraction=max(0.0, policy.probation_fraction - 0.25)),
                fc,
                e,
                replace(policy, probation_fraction=min(1.0, policy.probation_fraction + 0.25)),
                fc,
            )
        )
    if fc.enabled:
        specs.extend(
            [
                (
                    "Capital access of constrained students (±50%)",
                    e,
                    policy,
                    replace(
                        fc,
                        constrained_cap_access=max(0.0, 0.5 * fc.constrained_cap_access),
                    ),
                    e,
                    policy,
                    replace(
                        fc,
                        constrained_cap_access=min(1.0, 1.5 * fc.constrained_cap_access),
                    ),
                ),
                (
                    "Absorbable unfunded gap (±50%)",
                    e,
                    policy,
                    replace(
                        fc,
                        absorbable_gap_base=0.5 * fc.absorbable_gap_base,
                        absorbable_gap_poverty=0.5 * fc.absorbable_gap_poverty,
                    ),
                    e,
                    policy,
                    replace(
                        fc,
                        absorbable_gap_base=1.5 * fc.absorbable_gap_base,
                        absorbable_gap_poverty=1.5 * fc.absorbable_gap_poverty,
                    ),
                ),
                (
                    "Unfunded-gap retention penalty (±50%)",
                    e,
                    policy,
                    replace(
                        fc,
                        gap_retention_penalty=max(0.0, 0.5 * fc.gap_retention_penalty),
                    ),
                    e,
                    policy,
                    replace(
                        fc,
                        gap_retention_penalty=min(0.5, 1.5 * fc.gap_retention_penalty),
                    ),
                ),
                (
                    "Work-hours cap (±5 hours)",
                    e,
                    policy,
                    replace(fc, work_hours_cap=max(1.0, fc.work_hours_cap - 5.0)),
                    e,
                    policy,
                    replace(fc, work_hours_cap=fc.work_hours_cap + 5.0),
                ),
            ]
        )

    rows = []
    for label, e_low, p_low, f_low, e_high, p_high, f_high in specs:
        low = retained_delta(e_low, p_low, f_low)
        high = retained_delta(e_high, p_high, f_high)
        rows.append(
            {
                "assumption": label,
                "low_delta": low,
                "high_delta": high,
                "central_delta": central,
                "swing": abs(high - low),
            }
        )
    return pd.DataFrame(rows).sort_values("swing", ascending=True).reset_index(drop=True)


def seed_robustness(
    settings: CohortSettings,
    policy: Policy,
    e: Economics,
    n_seeds: int = 15,
    fc: Optional[FinancingConstraints] = None,
) -> pd.DataFrame:
    """Re-run policy comparison across independent synthetic random seeds."""
    if not (2 <= n_seeds <= 100):
        raise ValueError("Use between 2 and 100 random seeds.")
    rows = []
    for offset in range(n_seeds):
        try:
            cohort = build_synthetic_cohort(replace(settings, seed=settings.seed + offset))
        except ValueError:
            continue
        strict = evaluate_policy(cohort, STRICT_POLICY, e, fc)["metrics"]
        alt = evaluate_policy(cohort, policy, e, fc)["metrics"]
        rows.append(
            {
                "seed": settings.seed + offset,
                "delta_expected_retained": alt["expected_retained"] - strict["expected_retained"],
                "delta_retention_rate_pp": 100.0
                * (alt["expected_retention_rate"] - strict["expected_retention_rate"]),
                "delta_total_institutional_spend": alt["total_institutional_spend"]
                - strict["total_institutional_spend"],
                "delta_net_contribution": alt["expected_net_contribution"]
                - strict["expected_net_contribution"],
            }
        )
    return pd.DataFrame(rows)


def subgroup_summary(
    cohort: pd.DataFrame, baseline: dict, alternative: dict, min_cell: int = 15
) -> pd.DataFrame:
    """Summarize policy impacts across reliance, need, poverty proxy, and capital access."""
    need = cohort["need_index"].to_numpy(float)
    dependent = cohort["dependent"].to_numpy(int)
    strict_p = baseline["detail"]["retention_probability"].to_numpy(float)
    alt_p = alternative["detail"]["retention_probability"].to_numpy(float)
    alt_fraction = alternative["detail"]["award_fraction"].to_numpy(float)
    strict_fraction = baseline["detail"]["award_fraction"].to_numpy(float)
    alt_loans = alternative["detail"]["potential_added_loans"].to_numpy(float)
    alt_gap = alternative["detail"]["potential_uncovered_gap"].to_numpy(float)

    tertile_edges = np.quantile(need, [1.0 / 3.0, 2.0 / 3.0])
    need_group = np.select(
        [need <= tertile_edges[0], need <= tertile_edges[1]],
        ["Lowest-need third", "Middle-need third"],
        "Highest-need third",
    )
    reliance = np.where(
        dependent == 1, "Financially reliant on award", "Less reliant on award"
    )

    partitions: list[tuple[str, np.ndarray]] = [
        ("Award Reliance", reliance),
        ("Financial Need Tier", need_group),
    ]
    if "pell_eligible" in cohort.columns:
        partitions.append(
            (
                "Poverty Proxy",
                np.where(
                    cohort["pell_eligible"].to_numpy(int) == 1,
                    "Pell-proxy",
                    "Not Pell-proxy",
                ),
            )
        )
    if "capital_constrained" in cohort.columns:
        partitions.append(
            (
                "Capital Access",
                np.where(
                    cohort["capital_constrained"].to_numpy(int) == 1,
                    "Capital-constrained",
                    "Unconstrained",
                ),
            )
        )
    elif any(c in cohort.columns for c in ("maxed_federal_loans", "plus_denied")):
        rev_cols = [c for c in ("maxed_federal_loans", "plus_denied") if c in cohort.columns]
        rev_flag = np.column_stack([cohort[c].to_numpy(int) for c in rev_cols]).max(axis=1)
        partitions.append(
            (
                "Capital Access",
                np.where(rev_flag == 1, "Capital-constrained (revealed)", "Unconstrained"),
            )
        )

    frames = []
    for group_name, labels in partitions:
        for value in pd.unique(labels):
            mask = labels == value
            size = int(mask.sum())
            if size < min_cell:
                continue
            frames.append(
                {
                    "grouping": group_name,
                    "group": value,
                    "students": size,
                    "strict_retention_rate": float(strict_p[mask].mean()),
                    "alt_retention_rate": float(alt_p[mask].mean()),
                    "retention_change_pp": float(
                        100.0 * (alt_p[mask].mean() - strict_p[mask].mean())
                    ),
                    "strict_full_loss": int((strict_fraction[mask] < 1e-8).sum()),
                    "alt_full_loss": int((alt_fraction[mask] < 1e-8).sum()),
                    "mean_potential_added_loans": float(alt_loans[mask].mean()),
                    "mean_unfunded_gap": float(alt_gap[mask].mean()),
                }
            )
    return pd.DataFrame(frames)


def near_cutoff_diagnostic(
    cohort: pd.DataFrame,
    cutoff: float = 3.5,
    bandwidth: float = 0.20,
    min_each_side: int = 10,
) -> Optional[dict]:
    """Compute descriptive persistence averages near the 3.50 GPA cutoff."""
    if "reenrolled_next_term" not in cohort:
        return None
    below = cohort[
        (cohort.college_gpa >= cutoff - bandwidth) & (cohort.college_gpa < cutoff)
    ]
    above = cohort[
        (cohort.college_gpa >= cutoff) & (cohort.college_gpa <= cutoff + bandwidth)
    ]
    if len(below) < min_each_side or len(above) < min_each_side:
        return None
    result = {
        "n_below": len(below),
        "n_above": len(above),
        "retention_below": float(below.reenrolled_next_term.mean()),
        "retention_above": float(above.reenrolled_next_term.mean()),
        "difference_above_minus_below": float(
            above.reenrolled_next_term.mean() - below.reenrolled_next_term.mean()
        ),
    }
    if "renewed_actual" in cohort:
        result["award_renewal_below"] = float(below.renewed_actual.mean())
        result["award_renewal_above"] = float(above.renewed_actual.mean())
    return result


def section_grade_diagnostic(cohort: pd.DataFrame, min_cell: int = 15) -> pd.DataFrame:
    """Calculate OLS grade residuals by anonymous course section."""
    if "section_group" not in cohort:
        return pd.DataFrame()
    x_fields = ["hs_gpa"]
    if "readiness_score" in cohort:
        x_fields.append("readiness_score")
    X = np.column_stack(
        [np.ones(len(cohort))] + [cohort[c].to_numpy(float) for c in x_fields]
    )
    y = cohort.college_gpa.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    temp = pd.DataFrame(
        {
            "section": cohort.section_group.to_numpy(str),
            "residual": y - X @ beta,
            "college_gpa": y,
        }
    )
    table = temp.groupby("section", as_index=False).agg(
        students=("residual", "size"),
        mean_grade_residual=("residual", "mean"),
        average_college_gpa=("college_gpa", "mean"),
    )
    return (
        table.loc[table.students >= min_cell]
        .sort_values("mean_grade_residual", ascending=False)
        .reset_index(drop=True)
    )


# ===========================================================================
# PART 2 · LATEX-SAFE MARKDOWN ENGINE & CLEAN PLOTLY VISUALS
# ===========================================================================

COLORS = {
    "strict": "#C0392B",      # Brick crimson for strict 3.50 cutoff
    "full_keep": "#1F6F78",   # Deep teal for full renewal
    "probation": "#D48806",   # Warm amber for partial renewal
    "loss": "#94A3B8",        # Muted slate for lost award
    "navy": "#1E3A5F",        # Executive navy
    "low_bound": "#2B6CB0",   # Clean blue for lower sensitivity bound
    "high_bound": "#C05621",  # Burnt orange for upper sensitivity bound
}

STUDY_LINKS = {
    "NCES — HBCU Enrollment and Financial Aid Fast Facts": "https://nces.ed.gov/fastfacts/display.asp?id=667",
    "ACT Research — Grade Inflation and Readiness Implications (2025)": "https://www.act.org/content/dam/act/unsecured/documents/R2432-The-Implications-of-Grade-Inflation-2025-11.pdf",
    "U.S. Department of Education — College Scorecard Data": "https://collegescorecard.ed.gov/data/",
    "Carruthers & Özek — Losing HOPE: Aid Loss and Student Employment": "https://www.sciencedirect.com/science/article/pii/S027277571630173X",
    "Performance Standards and Financial Aid Persistence": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10923567/",
    "Stinebrickner & Stinebrickner (2008) — Credit Constraints and College Drop-Out": "https://www.nber.org/papers/w13340",
    "Lochner & Monge-Naranjo — Credit Constraints in Education (Review)": "https://www.nber.org/papers/w17435",
    "Denning, Marx & Turner — Pell Grants, Graduation, and Earnings (ProPelled)": "https://www.aeaweb.org/doi/10.1257/app.20180100",
    "Marx & Turner — Student Loan Access and Attainment (Field Experiment)": "https://www.aeaweb.org/articles?id=10.1257/pol.20180279",
    "Denning & Jones — Maxed Out? Larger Student Loan Limits": "https://docs.iza.org/dp12239.pdf",
    "Stinebrickner & Stinebrickner (2003) — Working During School and Grades": "https://econpapers.repec.org/RePEc:ucp:jlabec:v:21:y:2003:i:2:p:449-472",
    "Hope Center — Student Basic Needs Survey (#RealCollegeHBCU Context)": "https://hope.temple.edu/sites/hope/files/media/document/Hope%20Student%20Basic%20Needs%20Survey%20Report%202023-24.pdf",
    "U.S. Department of Education — FERPA Privacy Guidance": "https://studentprivacy.ed.gov/ferpa",
}


# --- Currency Formatters ---
# Plain formatters (for st.metric, st.dataframe, Plotly labels, and raw file downloads)
def fmt_usd(val: float, decimals: int = 0) -> str:
    if val < 0:
        return f"-${abs(val):,.{decimals}f}"
    return f"${val:,.{decimals}f}"


def fmt_signed_usd(val: float, decimals: int = 0) -> str:
    sign = "+" if val >= 0 else "-"
    return f"{sign}${abs(val):,.{decimals}f}"


# Markdown-safe formatters (escape '$' as '\$' so Streamlit NEVER triggers LaTeX math mode)
def md_usd(val: float, decimals: int = 0) -> str:
    if val < 0:
        return f"-\\${abs(val):,.{decimals}f}"
    return f"\\${val:,.{decimals}f}"


def md_signed_usd(val: float, decimals: int = 0) -> str:
    sign = "+" if val >= 0 else "-"
    return f"{sign}\\${abs(val):,.{decimals}f}"


def fmt_pct(val: float) -> str:
    return f"{100.0 * val:.1f}%"


def escape_dollar_for_streamlit(text: str) -> str:
    """Escape bare currency '$' in AI-generated text before st.markdown rendering."""
    out = []
    for i, ch in enumerate(text):
        if ch == "$" and (i == 0 or text[i - 1] != "\\"):
            out.append("\\$")
        else:
            out.append(ch)
    return "".join(out)


def apply_clean_theme_css() -> None:
    """Minimal, theme-adaptive styling that respects both Light and Dark mode."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }
        div[data-testid="stMetric"] {
            padding: 0.35rem 0.2rem;
        }
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- Plain-English Dynamic Markdown Builders ---------------- #


def explain_headline_summary(
    b: dict,
    a: dict,
    delta: dict,
    policy: Policy,
    econ: Economics,
    fc: FinancingConstraints,
    for_streamlit: bool = True,
) -> tuple[str, str]:
    """Return (status_type, clear_markdown_summary)."""
    u = md_usd if for_streamlit else fmt_usd
    su = md_signed_usd if for_streamlit else fmt_signed_usd

    d_ret = delta["delta_expected_retained"]
    d_grad = delta["delta_expected_completion_proxy"]
    d_spend = delta["delta_total_institutional_spend"]
    d_net = delta["delta_expected_net_contribution"]
    net_tuition_before_aid = econ.tuition_per_retained - econ.marginal_instruction_cost
    fc_tag = " *(with Poverty & Capitalization Constraints ON)*" if fc.enabled else ""

    if abs(d_ret) < 0.05 and abs(d_spend) < 50:
        return (
            "info",
            f"**Policy Summary — Matches Strict 3.50 Baseline{fc_tag}:** Your alternative policy is currently identical "
            "to the strict 3.50 GPA rule. Adjust the **Alternative Policy Design** sliders in the left sidebar "
            "to test a lower renewal cutoff, a partial-aid probation window, or early-warning tutoring.",
        )

    if d_ret > 0 and d_net >= 0:
        return (
            "success",
            f"**Policy Summary — Higher Retention That Pays for Itself{fc_tag}:**\n"
            f"- **Enrollment impact:** Keeps **{d_ret:+.1f} additional expected students** enrolled next year "
            f"(and adds **{d_grad:+.1f}** expected multi-year graduates).\n"
            f"- **Aid & tutoring outlay:** Commits **{su(d_spend)}** more in scholarship renewals and academic support.\n"
            f"- **Net university budget effect:** Improves net operating contribution by **{su(d_net)}**, because each "
            f"returning student generates **{u(net_tuition_before_aid)}** in tuition above marginal instructional cost.",
        )

    if d_ret > 0 and d_net < 0:
        gross_per_student = d_spend / d_ret
        net_cost_per_student = abs(d_net) / d_ret
        return (
            "warning",
            f"**Policy Summary — Higher Student Retention with a Net Budget Cost{fc_tag}:**\n"
            f"- **Enrollment impact:** Keeps **{d_ret:+.1f} additional expected students** enrolled next year "
            f"(and adds **{d_grad:+.1f}** expected multi-year graduates).\n"
            f"- **Gross aid & tutoring outlay:** Increases scholarship renewals and tutoring spend by **{su(d_spend)}** "
            f"(**{u(gross_per_student)}** per additional returning student).\n"
            f"- **Net university budget cost:** After crediting the tuition paid by those returning students, the net annual "
            f"cost to the institution is **{u(abs(d_net))}** (**{u(net_cost_per_student)}** net subsidy per added returner).",
        )

    if d_ret < 0 and d_net > 0:
        return (
            "error",
            f"**Policy Summary — Short-Term Aid Savings at the Cost of Student Attrition{fc_tag}:**\n"
            f"- **Enrollment impact:** Reduces expected returning enrollment by **{abs(d_ret):.1f} students**.\n"
            f"- **Budget impact:** Lowers scholarship outlays by **{u(abs(d_spend))}** and raises short-term net tuition "
            f"contribution by **{su(d_net)}**, but shrinks the continuing student body.",
        )

    return (
        "error",
        f"**Policy Summary — Lower Enrollment and Lower Net Contribution{fc_tag}:**\n"
        f"- **Enrollment impact:** Expected returning headcount changes by **{d_ret:+.1f} students**.\n"
        f"- **Budget impact:** Net institutional tuition contribution changes by **{su(d_net)}**.",
    )


def explain_eligibility_shifts(b: dict, a: dict, policy: Policy) -> str:
    """Clear, LaTeX-safe bullet breakdown of GPA cutoff mechanics."""
    n = b["n"]
    spared = b["full_loss_count"] - a["full_loss_count"]

    lines = [
        f"- **Strict 3.50 baseline:** **{b['full_loss_count']:,} of {n:,} recipients ({100*b['full_loss_count']/n:.1f}%)** earn below 3.50 and forfeit 100% of their scholarship—including **{b['dependent_full_loss_count']:,}** students modeled as financially reliant on the award.",
        f"- **Alternative ({policy.threshold:.2f} cutoff):** **{a['full_renewal_count']:,} students** keep 100% of their award, **{a['partial_renewal_count']:,} students** retain a **{policy.probation_fraction:.0%} partial award** on probation, and **{a['full_loss_count']:,} students** lose the full award.",
    ]
    if spared > 0:
        lines.append(
            f"- **Cliff reduction:** **{spared:,} fewer students** experience an abrupt, all-or-nothing scholarship cancellation."
        )
    elif spared < 0:
        lines.append(
            f"- **Stricter cliff:** **{abs(spared):,} additional students** lose 100% of their scholarship compared with the 3.50 baseline."
        )
    if a["targeted_count"] > 0:
        lines.append(
            f"- **Early-warning tutoring:** **{a['targeted_count']:,} near-cutoff students** receive academic coaching (direct outlay: **{md_usd(a['support_spend'])}**), lifting **{a['rescued_to_full_count']:,} students** over the **{policy.threshold:.2f}** full-renewal threshold under the assumed **+{policy.support_gpa_lift:.2f} GPA** gain."
        )
    lines.append(
        "- **Why headcounts show decimals:** Expected retention sums each student's individual probability of returning (e.g., 0.82), producing an exact statistical expectation across the cohort."
    )
    return "\n".join(lines)


def explain_student_finances(
    b: dict, a: dict, econ: Economics, fc: FinancingConstraints
) -> str:
    """Clear, LaTeX-safe explanation of student work hours, borrowing, and constraints."""
    unmet_pct = max(0.0, 1.0 - econ.work_replacement_share - econ.loan_replacement_share)
    if a["affected_any_loss_count"] == 0:
        return "- **Zero lost aid:** Every recipient retains 100% scholarship funding under these settings, creating no added student loan debt or off-campus work hours."

    lines = [
        f"- **Total scholarship dollars lost:** **{a['affected_any_loss_count']:,} students** lose part or all of their award under the alternative rule, totaling **{md_usd(a['potential_lost_award_total'])}** in lost aid if all affected students reenroll (versus **{md_usd(b['potential_lost_award_total'])}** under the strict 3.50 rule).",
        f"- **Target household split:** Unconstrained students aim to replace **{econ.work_replacement_share:.0%}** of lost aid with job earnings, **{econ.loan_replacement_share:.0%}** with incremental student loans, and leave **{unmet_pct:.0%}** uncovered.",
        f"- **Individual student workload & debt:** At **{md_usd(econ.hourly_wage, 2)}/hour** across **{econ.academic_work_weeks} academic weeks**, an affected student who stays enrolled averages **{a['mean_work_hours_per_affected_student']:.1f} extra work hours per week** and **{md_usd(a['mean_loan_per_affected_student'])}** in extra annual borrowing.",
        f"- **Expected totals among returners:** Weighting by each student's probability of actually returning yields **{a['expected_extra_work_hours_week_among_reenrollees']:,.1f} combined weekly work hours** and **{md_usd(a['expected_extra_borrowing_among_reenrollees'])}** in added borrowing among returning students.",
    ]
    if fc.enabled:
        lines.append(
            f"- **Poverty & capitalization limits active:** Extra work is capped at **{fc.work_hours_cap:.0f} hrs/week**, and the **{a['capital_constrained_count']:,}** capital-constrained recipients reach only **{fc.constrained_cap_access:.0%}** of the loan channel (up to **{md_usd(fc.annual_loan_cap)}**/yr). Unearned/unborrowed dollars spill into the unpaid shortfall—pushing **{a['students_over_absorbable_gap']:,} students** past their absorbable buffer (**{md_usd(fc.absorbable_gap_poverty)}** for poverty-proxy households vs. **{md_usd(fc.absorbable_gap_base)}** otherwise) and adding up to **{100*fc.gap_retention_penalty:.0f} percentage points** of extra stop-out risk."
        )
    return "\n".join(lines)


def explain_budget_ledger(a: dict, econ: Economics) -> str:
    """Explain Aid Outlay vs. Net Tuition Contribution with escaped currency."""
    return (
        f"**Understanding the two budget metrics:**\n"
        f"- **Scholarship + Tutoring Outlay ({md_usd(a['total_institutional_spend'])}):** Total institutional aid discounts awarded to returning students plus direct tutoring expenditures.\n"
        f"- **Net Tuition Contribution ({md_usd(a['expected_net_contribution'])}):** Net revenue remaining for general university operations after taking gross tuition (**{md_usd(econ.tuition_per_retained)}** per returner) and subtracting marginal instructional cost (**{md_usd(econ.marginal_instruction_cost)}** per returner), scholarship discounts, and tutoring spend."
    )


# ---------------- Clean, Non-Overlapping Plotly Charts ---------------- #


def _style_chart(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply clean, transparent layout compatible with Light and Dark themes."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12.5),
        title_font=dict(size=15),
        margin=dict(l=16, r=20, t=58, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.22)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.22)", zeroline=False)
    return fig


def chart_gpa_distribution(cohort: pd.DataFrame, policy: Policy) -> go.Figure:
    """GPA histogram color-coded by renewal status under the alternative policy."""
    gpas = cohort.college_gpa.to_numpy()
    edges = np.linspace(0.0, 4.0, 41)
    centers = 0.5 * (edges[1:] + edges[:-1])

    full_counts, prob_counts, loss_counts = [], [], []
    for low_e, high_e, c in zip(edges[:-1], edges[1:], centers):
        in_bin = int(((gpas >= low_e) & (gpas < high_e if high_e < 4.0 else gpas <= high_e)).sum())
        if c >= policy.threshold - 1e-6:
            full_counts.append(in_bin)
            prob_counts.append(0)
            loss_counts.append(0)
        elif (
            policy.probation_band > 0
            and policy.probation_fraction > 0
            and c >= policy.threshold - policy.probation_band - 1e-6
        ):
            full_counts.append(0)
            prob_counts.append(in_bin)
            loss_counts.append(0)
        else:
            full_counts.append(0)
            prob_counts.append(0)
            loss_counts.append(in_bin)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=full_counts,
            width=0.088,
            name=f"Full Renewal (≥ {policy.threshold:.2f})",
            marker_color=COLORS["full_keep"],
            hovertemplate="GPA ~ %{x:.2f}<br>Full renewal: %{y} students<extra></extra>",
        )
    )
    if sum(prob_counts) > 0:
        fig.add_trace(
            go.Bar(
                x=centers,
                y=prob_counts,
                width=0.088,
                name=f"Probation ({policy.probation_fraction:.0%} Aid)",
                marker_color=COLORS["probation"],
                hovertemplate="GPA ~ %{x:.2f}<br>Probation partial aid: %{y} students<extra></extra>",
            )
        )
    fig.add_trace(
        go.Bar(
            x=centers,
            y=loss_counts,
            width=0.088,
            name="Full Aid Loss ($0)",
            marker_color=COLORS["loss"],
            hovertemplate="GPA ~ %{x:.2f}<br>Full aid loss: %{y} students<extra></extra>",
        )
    )

    fig.add_vline(
        x=3.50,
        line_dash="dash",
        line_width=2.2,
        line_color=COLORS["strict"],
        annotation_text="Strict 3.50",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["strict"], size=11.5),
    )
    if abs(policy.threshold - 3.50) > 0.08:
        fig.add_vline(
            x=policy.threshold,
            line_dash="dot",
            line_width=2.2,
            line_color=COLORS["full_keep"],
            annotation_text=f"Alt {policy.threshold:.2f}",
            annotation_position="top left",
            annotation_font=dict(color=COLORS["full_keep"], size=11.5),
        )
    elif abs(policy.threshold - 3.50) > 1e-5:
        fig.add_vline(
            x=policy.threshold,
            line_dash="dot",
            line_width=2.2,
            line_color=COLORS["full_keep"],
        )

    xmin = max(1.6, float(np.floor(gpas.min() * 4.0) / 4.0) - 0.15)
    fig.update_layout(
        barmode="stack",
        title="First-Year College GPA Distribution by Pre-Tutoring Award Tier",
        xaxis_title="First-Year College GPA (0.00–4.00 Scale)",
        yaxis_title="Number of Students",
        xaxis_range=[xmin, 4.05],
    )
    return _style_chart(fig, height=390)


def chart_financing_breakdown(b: dict, a: dict) -> go.Figure:
    """Stacked horizontal bar comparing how lost scholarship dollars are absorbed."""
    labels = ["Alternative Policy", "Strict 3.50 Rule"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[a["potential_work_replacement_dollars"], b["potential_work_replacement_dollars"]],
            name="Covered by Student Wages",
            orientation="h",
            marker_color=COLORS["full_keep"],
            hovertemplate="%{y}<br>Covered by job wages: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[a["potential_extra_borrowing_total"], b["potential_extra_borrowing_total"]],
            name="Covered by Extra Loans",
            orientation="h",
            marker_color=COLORS["probation"],
            hovertemplate="%{y}<br>Covered by student loans: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[a["potential_uncovered_gap_total"], b["potential_uncovered_gap_total"]],
            name="Unpaid Cash Shortfall",
            orientation="h",
            marker_color=COLORS["strict"],
            hovertemplate="%{y}<br>Unpaid cash shortfall: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        title="How Lost Scholarship Dollars Hit Student Budgets (Cohort Total)",
        xaxis_title="Total Lost Award Dollars ($)",
        yaxis_title=None,
    )
    return _style_chart(fig, height=265)


def chart_optimization_frontier(
    candidates: pd.DataFrame,
    selected: pd.Series,
    budget: float,
    objective_key: str,
    objective_label: str,
) -> go.Figure:
    """Scatter plot of feasible vs. over-budget policy designs with Pareto frontier."""
    feasible = candidates[candidates.additional_spend <= budget + 0.01]
    infeasible = candidates[candidates.additional_spend > budget + 0.01]

    sorted_all = candidates.sort_values(["additional_spend", objective_key], ascending=[True, False])
    pareto = []
    best_so_far = -np.inf
    for _, r in sorted_all.iterrows():
        if r[objective_key] > best_so_far + 1e-9:
            pareto.append(r)
            best_so_far = r[objective_key]
    pareto_df = pd.DataFrame(pareto)

    fig = go.Figure()
    if not infeasible.empty:
        fig.add_trace(
            go.Scatter(
                x=infeasible.additional_spend,
                y=infeasible[objective_key],
                mode="markers",
                name="Over Budget Cap",
                opacity=0.35,
                marker=dict(size=8, color=COLORS["loss"]),
                customdata=np.stack(
                    [infeasible.renewal_gpa, infeasible.probation_award_pct, infeasible.pre_renewal_support],
                    axis=-1,
                ),
                hovertemplate=(
                    "Cutoff: %{customdata[0]:.2f} | Probation: %{customdata[1]:.0f}% | Tutoring: %{customdata[2]}"
                    "<br>Added Spend: $%{x:,.0f}<br>Gain vs. 3.50: %{y:,.1f}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=feasible.additional_spend,
            y=feasible[objective_key],
            mode="markers",
            name="Within Budget Cap",
            opacity=0.85,
            marker=dict(size=8.5, color=COLORS["full_keep"]),
            customdata=np.stack(
                [feasible.renewal_gpa, feasible.probation_award_pct, feasible.pre_renewal_support],
                axis=-1,
            ),
            hovertemplate=(
                "Cutoff: %{customdata[0]:.2f} | Probation: %{customdata[1]:.0f}% | Tutoring: %{customdata[2]}"
                "<br>Added Spend: $%{x:,.0f}<br>Gain vs. 3.50: %{y:,.1f}<extra></extra>"
            ),
        )
    )

    if len(pareto_df) > 1:
        fig.add_trace(
            go.Scatter(
                x=pareto_df.additional_spend,
                y=pareto_df[objective_key],
                mode="lines",
                name="Best-Per-Dollar Frontier",
                line=dict(color=COLORS["navy"], width=1.8, dash="dot"),
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[selected.additional_spend],
            y=[selected[objective_key]],
            mode="markers",
            name="Recommended Policy",
            marker=dict(symbol="star", size=19, color=COLORS["probation"], line=dict(width=1.5, color="#333333")),
            hovertemplate=(
                f"Selected: {selected.renewal_gpa:.2f} GPA ({selected.probation_award_pct:.0f}% probation)"
                "<br>Added Spend: $%{x:,.0f}<br>Gain vs. 3.50: %{y:,.1f}<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=budget,
        line_dash="dash",
        line_width=2,
        line_color=COLORS["strict"],
        annotation_text=f"Cap: {fmt_usd(budget)}",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["strict"]),
    )

    fig.update_layout(
        title=f"Policy Tradeoff Frontier: {objective_label} vs. Added Aid Spend",
        xaxis_title="Added Annual Scholarship + Tutoring Spend vs. Strict 3.50 ($)",
        yaxis_title=objective_label,
    )
    return _style_chart(fig, height=420)


def chart_tornado(sensitivity: pd.DataFrame) -> go.Figure:
    """Horizontal tornado chart showing which assumption swings retention most."""
    central = float(sensitivity.central_delta.iloc[0])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=sensitivity.assumption,
            x=sensitivity.low_delta - central,
            base=central,
            orientation="h",
            name="Lower Assumption Value",
            marker_color=COLORS["low_bound"],
            text=[f"{v:+.1f}" for v in sensitivity.low_delta],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>At lower value: %{text} students<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=sensitivity.assumption,
            x=sensitivity.high_delta - central,
            base=central,
            orientation="h",
            name="Higher Assumption Value",
            marker_color=COLORS["high_bound"],
            text=[f"{v:+.1f}" for v in sensitivity.high_delta],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>At higher value: %{text} students<extra></extra>",
        )
    )
    fig.add_vline(
        x=central,
        line_dash="dash",
        line_width=2,
        line_color=COLORS["navy"],
        annotation_text=f"Central: {central:+.1f}",
        annotation_position="top",
    )
    fig.update_layout(
        barmode="overlay",
        title="Assumption Sensitivity: How Much Does Expected Reenrollment Swing?",
        xaxis_title="Added Expected Returning Students (Alternative minus Strict 3.50)",
        yaxis_title=None,
        margin=dict(l=16, r=60, t=58, b=18),
    )
    return _style_chart(fig, height=150 + 62 * len(sensitivity))


def chart_threshold_response(sweep: pd.DataFrame, current_threshold: float) -> go.Figure:
    """Dual-axis curve showing Expected Reenrollment and Net Contribution across GPA cutoffs."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=sweep.threshold,
            y=sweep.expected_retained,
            mode="lines+markers",
            name="Expected Returning Students (Left)",
            line=dict(color=COLORS["full_keep"], width=2.5),
            marker=dict(size=6),
            customdata=np.stack([sweep.total_institutional_spend, sweep.full_loss_count], axis=-1),
            hovertemplate=(
                "GPA Cutoff: %{x:.2f}<br>Expected Returning: %{y:.1f} students"
                "<br>Total Aid + Tutoring Spend: $%{customdata[0]:,.0f}"
                "<br>Students Losing Full Aid: %{customdata[1]:,}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sweep.threshold,
            y=sweep.expected_net_contribution,
            mode="lines",
            name="Net Tuition Contribution $ (Right)",
            line=dict(color=COLORS["probation"], width=2.2, dash="dot"),
            hovertemplate="GPA Cutoff: %{x:.2f}<br>Net Tuition Contribution: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_vline(
        x=3.50,
        line_dash="dash",
        line_color=COLORS["strict"],
        annotation_text="Strict 3.50",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["strict"]),
    )
    if abs(current_threshold - 3.50) > 0.08:
        fig.add_vline(
            x=current_threshold,
            line_dash="dot",
            line_color=COLORS["full_keep"],
            annotation_text=f"Alt {current_threshold:.2f}",
            annotation_position="top left",
            annotation_font=dict(color=COLORS["full_keep"]),
        )

    fig.update_layout(
        title="Enrollment vs. Net Tuition Contribution Across GPA Cutoffs (2.50–3.80)",
        xaxis_title="Full-Renewal College GPA Cutoff",
    )
    fig.update_yaxes(title_text="Expected Returning Students", secondary_y=False)
    fig.update_yaxes(
        title_text="Net Tuition Contribution ($)",
        secondary_y=True,
        showgrid=False,
        tickprefix="$",
    )
    return _style_chart(fig, height=400)


def chart_readiness_quintiles(cohort: pd.DataFrame) -> Optional[go.Figure]:
    """Grouped bar comparing Entry HS GPA and First-Year College GPA across readiness fifths."""
    if "readiness_score" not in cohort.columns:
        return None
    s = cohort.copy()
    s["quintile"] = pd.qcut(
        s.readiness_score.rank(method="first"),
        q=5,
        labels=["Lowest 20%", "Second 20%", "Middle 20%", "Fourth 20%", "Highest 20%"],
    )
    g = (
        s.groupby("quintile", observed=True)
        .agg(hs_gpa=("hs_gpa", "mean"), college_gpa=("college_gpa", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=g.quintile,
            y=g.hs_gpa,
            name="Entry High-School GPA",
            marker_color=COLORS["navy"],
            text=[f"{v:.2f}" for v in g.hs_gpa],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=g.quintile,
            y=g.college_gpa,
            name="First-Year College GPA",
            marker_color=COLORS["full_keep"],
            text=[f"{v:.2f}" for v in g.college_gpa],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=3.50,
        line_dash="dash",
        line_color=COLORS["strict"],
        annotation_text="3.50 Cutoff",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="group",
        title="Entry High-School GPA vs. First-Year College GPA by Readiness Group",
        xaxis_title="Entering Academic Readiness Group (Quintiles)",
        yaxis_title="Average GPA (4.00 Scale)",
        yaxis_range=[0.0, 4.35],
    )
    return _style_chart(fig, height=385)


# ---------------- Policy Brief & Export Payload Builders ---------------- #


def build_policy_memo(
    mode: str,
    baseline: dict,
    alt: dict,
    delta: dict,
    policy: Policy,
    economics: Economics,
    cohort: pd.DataFrame,
    fc: FinancingConstraints,
    for_streamlit: bool = True,
) -> str:
    """Create a clean Administration Brief (LaTeX-safe on screen, clean Markdown on download)."""
    u = md_usd if for_streamlit else fmt_usd
    su = md_signed_usd if for_streamlit else fmt_signed_usd

    b, a = baseline["metrics"], alt["metrics"]
    completion_delta = delta["delta_expected_completion_proxy"]
    pv_earnings = (
        completion_delta
        * economics.earnings_premium_per_year
        * ten_year_annuity_factor(economics)
    )
    source_note = (
        "synthetic simulation data"
        if mode == "Synthetic cohort"
        else "de-identified uploaded records combined with uncalibrated behavioral assumptions"
    )
    _, plain_summary = explain_headline_summary(
        b, a, delta, policy, economics, fc, for_streamlit=for_streamlit
    )

    if fc.enabled:
        constraints_bullet = (
            f"- **Poverty & Capitalization Constraints (ENABLED):** Capital-constrained recipients "
            f"(**{a['capital_constrained_count']:,}** students) reach **{fc.constrained_cap_access:.0%}** of the "
            f"loan-replacement channel (annual loan cap: **{u(fc.annual_loan_cap)}**), incremental student work is "
            f"capped at **{fc.work_hours_cap:.0f} hours/week**, and unpaid shortfalls exceeding **{u(fc.absorbable_gap_base)}** "
            f"(or **{u(fc.absorbable_gap_poverty)}** for **{a['poverty_proxy_count']:,}** poverty-proxy recipients) impose "
            f"up to **{100.0 * fc.gap_retention_penalty:.0f} percentage points** of extra stop-out risk. Under the proposed "
            f"policy, **{a['students_over_absorbable_gap']:,} students** exceed their absorbable buffer (total excess gap: "
            f"**{u(a['excess_gap_total'])}**; change vs. strict 3.50: **{su(delta['delta_excess_gap_total'])}**). Flags reflect "
            f"de-identified Pell/loan-cap/PLUS-denial indicators where supplied, or need-rank imputations otherwise.\n"
        )
    else:
        constraints_bullet = (
            "- **Poverty & Capitalization Constraints (OFF):** Unconstrained baseline assumes all affected students "
            "can access the full work and loan replacement shares without individual hour or credit caps.\n"
        )

    return f"""### Scholarship Renewal Policy — Exploratory Administration Brief

**Governance & Evidence Status:** Hypothetical scenario projection computed from {source_note}. Figures below are **not** causal treatment-effect estimates, and institutional scholarship rules are assumed for stress-testing rather than independently verified.

#### 1. Executive Summary

{plain_summary}

#### 2. Policy Rules Compared ({len(cohort):,} Entering Scholarship Recipients)

- **Strict 3.50 Baseline:** Recipients must maintain at least a **3.50 first-year college GPA** to renew 100% of their award. Students below 3.50 forfeit the entire scholarship, with no probationary grace window or targeted pre-renewal coaching.
- **Proposed Alternative:** Full renewal at a **{policy.threshold:.2f} college GPA**; students within **{policy.probation_band:.2f} GPA points** below the cutoff retain **{policy.probation_fraction:.0%}** of their award on probation; proactive academic coaching covers students within **{policy.support_band:.2f} GPA points** below the cutoff (assuming a **+{policy.support_gpa_lift:.2f} GPA** gain at **{u(policy.support_cost_per_student)}** per flagged student).

#### 3. Model-Implied Annual Cohort & Budget Outcomes

| Outcome Measure | Strict 3.50 Rule | Proposed Alternative | Difference |
|---|---:|---:|---:|
| Students keeping 100% scholarship | {b['full_renewal_count']:,} | {a['full_renewal_count']:,} | {a['full_renewal_count'] - b['full_renewal_count']:+,} |
| Students on partial probation award | {b['partial_renewal_count']:,} | {a['partial_renewal_count']:,} | {a['partial_renewal_count'] - b['partial_renewal_count']:+,} |
| Students losing 100% of scholarship | {b['full_loss_count']:,} | {a['full_loss_count']:,} | {a['full_loss_count'] - b['full_loss_count']:+,} |
| Expected returning students | {b['expected_retained']:,.1f} | {a['expected_retained']:,.1f} | {delta['delta_expected_retained']:+,.1f} |
| Expected retention rate | {b['expected_retention_rate']:.1%} | {a['expected_retention_rate']:.1%} | {100.0 * delta['delta_expected_retention_rate']:+.1f} pp |
| Multi-year graduation proxy (students) | {b['expected_completion_proxy']:,.1f} | {a['expected_completion_proxy']:,.1f} | {completion_delta:+,.1f} |
| Expected gross tuition billed | {u(b['expected_gross_tuition'])} | {u(a['expected_gross_tuition'])} | {su(delta['delta_expected_gross_tuition'])} |
| Expected scholarship + tutoring spend | {u(b['total_institutional_spend'])} | {u(a['total_institutional_spend'])} | {su(delta['delta_total_institutional_spend'])} |
| Expected net tuition contribution | {u(b['expected_net_contribution'])} | {u(a['expected_net_contribution'])} | {su(delta['delta_expected_net_contribution'])} |
| Potential extra student loans (if all persist) | {u(b['potential_extra_borrowing_total'])} | {u(a['potential_extra_borrowing_total'])} | {su(delta['delta_potential_extra_borrowing_total'])} |
| Potential unpaid cash shortfall | {u(b['potential_uncovered_gap_total'])} | {u(a['potential_uncovered_gap_total'])} | {su(delta['delta_potential_uncovered_gap_total'])} |

#### 4. Key Behavioral & Financial Assumptions

- **Assumed Aid-Loss Retention Penalty:** Losing 100% of an award is assumed to reduce a financially reliant, midrange-need recipient's return probability by **{100.0 * economics.full_loss_retention_penalty:.0f} percentage points** prior to campus calibration.
- **Student Work & Borrowing Substitution:** Affected students aim to replace **{economics.work_replacement_share:.0%}** of lost aid via wages (at **{u(economics.hourly_wage, 2)}/hour**) and **{economics.loan_replacement_share:.0%}** via incremental student loans.
{constraints_bullet}- **Exploratory Graduate Earnings Exposure:** Over a {economics.earnings_horizon_years}-year horizon discounted at **{economics.real_discount_rate:.1%}** with an assumed **{u(economics.earnings_premium_per_year)}** annual completion earnings premium, the change in graduation-proxy outcomes represents **{u(pv_earnings)}** in illustrative present-value graduate earnings exposure (not an institutional accounting line item).

#### 5. Recommended Institutional Research Protocol

1. **Assemble Longitudinal Cohort Records (3–5 Years):** Link initial award terms, term GPAs, credit completion, appeal outcomes, unmet need, Pell/loan-limit/PLUS-denial indicators, and subsequent enrollment.
2. **Audit Section-Level Grading Dispersion:** Evaluate first-year grading variation across course sections after controlling for prior academic preparation and course rigor.
3. **Validate Cutoff Persistence Empirically:** Test for GPA bunching just above 3.50 and verify baseline covariate balance before interpreting near-cutoff persistence differences causally.
4. **Reconcile Endowment vs. Discount Accounting:** Separate donor-restricted cash awards from unfunded institutional tuition discounts, and distinguish campus merit rules from federal Satisfactory Academic Progress (SAP).
5. **Maintain FERPA Safeguards:** Enforce small-cell suppression ($n < 15$) and never use simulation outputs for automated individual aid decisions.
"""


def aggregate_payload(
    mode: str,
    baseline: dict,
    alternative: dict,
    delta: dict,
    alt_policy: Policy,
    econ: Economics,
    fc: FinancingConstraints,
) -> dict:
    """Aggregate-only JSON payload with zero student-level records."""
    safe_metrics = (
        "n",
        "expected_retained",
        "expected_retention_rate",
        "expected_completion_proxy",
        "full_loss_count",
        "partial_renewal_count",
        "dependent_full_loss_count",
        "total_institutional_spend",
        "expected_net_contribution",
        "potential_extra_borrowing_total",
        "potential_extra_work_hours_week_total",
        "potential_uncovered_gap_total",
        "poverty_proxy_count",
        "capital_constrained_count",
        "students_over_absorbable_gap",
        "excess_gap_total",
    )
    return {
        "data_type": (
            "synthetic"
            if mode == "Synthetic cohort"
            else "user-uploaded de-identified / unverified"
        ),
        "current_policy": asdict(STRICT_POLICY),
        "alternative_policy": asdict(alt_policy),
        "assumptions": asdict(econ),
        "financing_constraints": asdict(fc),
        "baseline_aggregate": {
            k: round(float(baseline["metrics"][k]), 3) for k in safe_metrics
        },
        "alternative_aggregate": {
            k: round(float(alternative["metrics"][k]), 3) for k in safe_metrics
        },
        "scenario_differences": {
            k: round(float(v), 3)
            for k, v in delta.items()
            if k
            in {
                "delta_expected_retained",
                "delta_total_institutional_spend",
                "delta_expected_net_contribution",
                "delta_expected_completion_proxy",
                "delta_students_over_absorbable_gap",
                "delta_excess_gap_total",
            }
        },
    }


def request_openai_interpretation(api_key: str, model_name: str, payload: dict) -> str:
    """Request plain-English teaching & administrative commentary from OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=30.0)
    instructions = (
        "You are an economics professor advising university leadership. "
        "Using clean, well-structured Markdown bullet points, write three short sections: "
        "1) What Moves in This Scenario and Why (including poverty/capitalization constraints if enabled); "
        "2) What Is Assumed vs. What Is Empirically Verified; "
        "3) Three Practical Questions for Campus Administrators. "
        "Do NOT use LaTeX math mode. Treat every number strictly as output from an uncalibrated "
        "scenario model, not a causal estimate. State clearly that scholarship rules and financing caps "
        "are assumed. Never invent facts, subgroup data, or citations, and never allege faculty grade inflation."
    )
    text_input = json.dumps(payload, sort_keys=True)

    if hasattr(client, "responses"):
        res = client.responses.create(
            model=model_name,
            store=False,
            max_output_tokens=900,
            instructions=instructions,
            input=text_input,
        )
        return res.output_text

    chat = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text_input},
        ],
        max_tokens=900,
        temperature=0.2,
    )
    return chat.choices[0].message.content or ""


# ===========================================================================
# PART 3 · STREAMLIT INTERFACE
# ===========================================================================

st.set_page_config(
    page_title="HBCU Scholarship Economics Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_clean_theme_css()


def key_from_env_or_secrets(name: str) -> str:
    val = os.getenv(name, "").strip()
    if val:
        return val
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def cached_cohort(settings: CohortSettings) -> pd.DataFrame:
    return build_synthetic_cohort(settings)


@st.cache_data(show_spinner=False)
def cached_optimization(
    cohort: pd.DataFrame,
    economics: Economics,
    fp: str,
    band: float,
    lift: float,
    support_band: float,
    unit_cost: float,
    budget: float,
    financing: FinancingConstraints,
) -> tuple[pd.DataFrame, pd.Series]:
    del fp
    base = evaluate_policy(cohort, STRICT_POLICY, economics, financing)
    return optimize_policies(
        cohort, economics, base, band, support_band, lift, unit_cost, budget, financing
    )


@st.cache_data(show_spinner=False)
def cached_sweep(
    cohort: pd.DataFrame,
    economics: Economics,
    band: float,
    fraction: float,
    support_band: float,
    lift: float,
    unit_cost: float,
    financing: FinancingConstraints,
) -> pd.DataFrame:
    return threshold_sweep(
        cohort,
        economics,
        band,
        fraction,
        support_band,
        lift,
        unit_cost,
        fc=financing,
    )


@st.cache_data(show_spinner=False)
def cached_seeds(
    settings: CohortSettings,
    policy: Policy,
    economics: Economics,
    n_seeds: int,
    financing: FinancingConstraints,
) -> pd.DataFrame:
    return seed_robustness(settings, policy, economics, n_seeds, financing)


# ---------------- Sidebar Configuration ---------------- #

with st.sidebar:
    st.subheader("1. Student Cohort Data")
    mode = st.radio(
        "Cohort Source",
        ["Synthetic cohort", "Approved de-identified CSV"],
        index=0,
    )
    cohort_settings: Optional[CohortSettings] = None

    if mode == "Synthetic cohort":
        n = st.slider("Entering scholarship students", 100, 5_000, 650, step=50)
        award_size = st.number_input(
            "Annual scholarship amount ($)", 500, 40_000, 8_000, step=500
        )
        dependent_share = st.slider(
            "Share reliant on award to stay enrolled", 0.00, 1.00, 0.70, step=0.05
        )
        with st.expander("Grading & Readiness Settings", expanded=False):
            entry_cutoff = st.slider(
                "Entry HS GPA cutoff", 2.50, 3.90, 3.50, step=0.05
            )
            hs_uplift = st.slider(
                "Assumed HS GPA upward shift", 0.00, 0.60, 0.20, step=0.05
            )
            hs_spread = st.slider(
                "HS grading variation across schools (SD)", 0.00, 0.40, 0.13, step=0.01
            )
            college_spread = st.slider(
                "College section grading variation (SD)", 0.00, 0.40, 0.14, step=0.01
            )
            college_uplift = st.slider(
                "Average college GPA shift", -0.25, 0.40, 0.00, step=0.05
            )
            seed = st.number_input("Random seed", 0, 100_000, 42)

        with st.expander("Financial Vulnerability Composition", expanded=False):
            poverty_share = st.slider(
                "Share at poverty proxy (Pell-like)", 0.00, 1.00, 0.55, step=0.05
            )
            constrained_share = st.slider(
                "Share capital-constrained (no wealth/credit buffer)",
                0.00,
                1.00,
                0.35,
                step=0.05,
            )
            st.caption(
                "Poverty models a household cash-flow constraint; capital access models a wealth/credit stock constraint. "
                "Both flags activate when the Poverty & Capitalization Constraints module (Section 3) is enabled."
            )

        cohort_settings = CohortSettings(
            n=int(n),
            seed=int(seed),
            entry_hs_cutoff=float(entry_cutoff),
            hs_grade_uplift=float(hs_uplift),
            hs_school_spread=float(hs_spread),
            college_section_spread=float(college_spread),
            college_grade_uplift=float(college_uplift),
            scholarship_amount=float(award_size),
            dependent_share=float(dependent_share),
            poverty_share=float(poverty_share),
            capital_constrained_share=float(constrained_share),
        )
        try:
            cohort = cached_cohort(cohort_settings)
        except ValueError as err:
            st.error(str(err))
            st.stop()
    else:
        st.caption(
            "Upload FERPA de-identified records only. Remove all names, student IDs, emails, and dates."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        template = pd.DataFrame(
            {
                "hs_gpa": [3.75, 3.85],
                "college_gpa": [3.40, 3.56],
                "scholarship_amount": [8000, 8000],
                "dependent": [1, 0],
                "need_index": [0.85, 0.35],
                "readiness_score": [63, 71],
                "section_group": ["SEC-01", "SEC-02"],
                "reenrolled_next_term": [1, 1],
                "renewed_actual": [0, 1],
                "pell_eligible": [1, 0],
                "maxed_federal_loans": [1, 0],
                "plus_denied": [0, 0],
            }
        )
        st.download_button(
            "Download CSV Schema Template",
            template.to_csv(index=False),
            "scholarship_upload_schema.csv",
            "text/csv",
            use_container_width=True,
        )
        if uploaded is None:
            st.info("Upload an approved CSV file in the sidebar, or select **Synthetic cohort**.")
            st.stop()
        try:
            raw = pd.read_csv(uploaded, low_memory=False)
            cohort = validate_uploaded_cohort(raw)
        except Exception as err:
            st.error(f"Upload rejected: {err}")
            st.stop()
        st.success(f"Loaded {len(cohort):,} de-identified records.")

    st.divider()
    st.subheader("2. Alternative Policy Design")
    renewal = st.slider("Full-renewal college GPA cutoff", 2.50, 3.80, 3.20, step=0.05)
    probation_band = st.slider(
        "Probation window below cutoff (GPA pts)", 0.00, 0.60, 0.25, step=0.05
    )
    probation_fraction = st.slider(
        "Portion of award kept on probation", 0.00, 1.00, 0.50, step=0.05
    )
    support_on = st.checkbox("Include early-warning tutoring", value=True)
    if support_on:
        support_band = st.slider(
            "Tutoring eligibility window below cutoff", 0.10, 0.70, 0.40, step=0.05
        )
        support_lift = st.slider(
            "Assumed GPA gain from tutoring", 0.00, 0.35, 0.10, step=0.01
        )
        support_cost = st.number_input(
            "Tutoring cost per student ($)", 0, 5_000, 450, step=50
        )
    else:
        support_band, support_lift, support_cost = 0.0, 0.0, 0.0

    st.divider()
    st.subheader("3. Budget & Behavior Assumptions")
    full_loss_impact_pp = st.slider(
        "Max retention drop from full aid loss (pp)",
        0,
        50,
        25,
        step=1,
        help="Assumed percentage-point drop in return rate when a financially reliant student loses 100% of the award.",
    )
    baseline_retention = st.slider(
        "Baseline return rate at 3.00 GPA", 0.40, 0.96, 0.82, step=0.01
    )
    gross_tuition = st.number_input(
        "Gross tuition per returning student ($)", 1_000, 50_000, 15_000, step=500
    )
    variable_cost = st.number_input(
        "Instructional cost per returning student ($)", 0, 50_000, 6_500, step=500
    )

    with st.expander("Work, Loans & Earnings Assumptions", expanded=False):
        work_fraction = st.slider(
            "Share of lost aid replaced by wages", 0.00, 1.00, 0.30, step=0.05
        )
        loan_fraction = st.slider(
            "Share of lost aid replaced by loans", 0.00, 1.00, 0.45, step=0.05
        )
        hourly_wage = st.number_input(
            "Assumed student hourly wage ($)", 7.0, 50.0, 15.0, step=0.5
        )
        earnings_premium = st.number_input(
            "Annual degree earnings premium ($)", 0, 80_000, 12_000, step=1_000
        )
        years = st.slider("Earnings horizon (years)", 1, 25, 10)
        real_rate = st.slider("Real discount rate", 0.000, 0.120, 0.035, step=0.005)

    if work_fraction + loan_fraction > 1.000001:
        st.error("Work share + Loan share cannot exceed 100%. Lower one in the sidebar.")
        st.stop()

    constraints_on = False
    cap_access_val, work_cap, loan_cap_val = 0.25, 15, 5_500
    gap_base, gap_pov, gap_pen_pp = 3_000, 750, 10
    impute_pov_share, impute_cap_share = 0.55, 0.35

    with st.expander("Poverty & Capitalization Constraints (Optional)", expanded=False):
        constraints_on = st.checkbox(
            "Model poverty & capital-access limits on replacing lost aid",
            value=False,
        )
        st.caption(
            "Off: every student can use the full work/loan shares above. "
            "On: caps weekly work hours, restricts borrowing for capital-constrained students, "
            "and penalizes persistence when unpaid gaps exceed household slack."
        )
        if constraints_on:
            cap_access_val = st.slider(
                "Loan-channel access for capital-constrained students",
                0.00,
                1.00,
                0.25,
                step=0.05,
                help="1.0 = full access to loan replacement share; 0.0 = maxed federal loans and no private/PLUS credit.",
            )
            work_cap = st.slider(
                "Max additional weekly work hours",
                1,
                40,
                15,
                help="Hours above this cap cannot be worked alongside full-time study; unearned dollars spill into the unpaid gap.",
            )
            loan_cap_val = st.number_input(
                "Annual incremental loan cap ($)",
                0,
                30_000,
                5_500,
                step=250,
            )
            gap_base = st.number_input(
                "Absorbable unpaid gap — non-poverty household ($)",
                0,
                20_000,
                3_000,
                step=250,
            )
            gap_pov = st.number_input(
                "Absorbable unpaid gap — poverty-proxy household ($)",
                0,
                10_000,
                750,
                step=250,
            )
            gap_pen_pp = st.slider(
                "Max extra retention drop from unabsorbable gap (pp)",
                0,
                40,
                10,
                help="Applied proportionally to unpaid gaps exceeding the household's absorbable buffer.",
            )
            if mode == "Approved de-identified CSV":
                has_pov_col = "pell_eligible" in cohort.columns
                has_cap_col = any(
                    c in cohort.columns
                    for c in ("capital_constrained", "maxed_federal_loans", "plus_denied")
                )
                if not has_pov_col:
                    impute_pov_share = st.slider(
                        "Imputed poverty-proxy share (no pell_eligible column)",
                        0.00,
                        1.00,
                        0.55,
                        step=0.05,
                    )
                if not has_cap_col:
                    impute_cap_share = st.slider(
                        "Imputed capital-constrained share (no flag columns)",
                        0.00,
                        1.00,
                        0.35,
                        step=0.05,
                    )
                if not (has_pov_col and has_cap_col):
                    st.warning(
                        "Missing flag columns are imputed by need_index rank. Supply de-identified "
                        "pell_eligible and capital_constrained (or maxed_federal_loans / plus_denied) for institutional analysis."
                    )

    financing = FinancingConstraints(
        enabled=bool(constraints_on),
        constrained_cap_access=float(cap_access_val),
        work_hours_cap=float(work_cap),
        annual_loan_cap=float(loan_cap_val),
        absorbable_gap_base=float(gap_base),
        absorbable_gap_poverty=float(gap_pov),
        gap_retention_penalty=float(gap_pen_pp) / 100.0,
        impute_poverty_share=float(impute_pov_share),
        impute_constrained_share=float(impute_cap_share),
    )

    economics = Economics(
        tuition_per_retained=float(gross_tuition),
        marginal_instruction_cost=float(variable_cost),
        baseline_retention=float(baseline_retention),
        full_loss_retention_penalty=float(full_loss_impact_pp) / 100.0,
        work_replacement_share=float(work_fraction),
        loan_replacement_share=float(loan_fraction),
        hourly_wage=float(hourly_wage),
        academic_work_weeks=32,
        earnings_premium_per_year=float(earnings_premium),
        earnings_horizon_years=int(years),
        real_discount_rate=float(real_rate),
    )
    alternative_policy = Policy(
        threshold=float(renewal),
        probation_band=float(probation_band),
        probation_fraction=float(probation_fraction),
        support_band=float(support_band),
        support_gpa_lift=float(support_lift),
        support_cost_per_student=float(support_cost),
    )

# Run Core Model
baseline = evaluate_policy(cohort, STRICT_POLICY, economics, financing)
alternative = evaluate_policy(cohort, alternative_policy, economics, financing)
delta = compare_policies(baseline, alternative)
b, a = baseline["metrics"], alternative["metrics"]

# ---------------- Clean Page Header ---------------- #

st.title("🎓 HBCU Scholarship Economics & Student Success Lab")
st.markdown(
    "Evaluate how a **strict 3.50 college-GPA renewal cliff** compares against adjusted GPA cutoffs, "
    "partial-aid probation windows, early-warning tutoring, and household liquidity constraints. All figures represent **scenario projections** "
    f"across **{len(cohort):,} {'synthetic' if mode == 'Synthetic cohort' else 'de-identified'} scholarship recipients**—not causal estimates at a specific institution."
)

# ---------------- Main Tabs ---------------- #

tab_lab, tab_opt, tab_sens, tab_meas, tab_brief, tab_guide = st.tabs(
    [
        "1. Policy Comparison",
        "2. Budget Optimizer",
        "3. Sensitivity & Equity",
        "4. Grading & Benchmarks",
        "5. Administration Brief",
        "6. Formulas & Lab Guide",
    ]
)

# ===========================================================================
# TAB 1 · POLICY COMPARISON
# ===========================================================================
with tab_lab:
    status_kind, summary_md = explain_headline_summary(
        b, a, delta, alternative_policy, economics, financing, for_streamlit=True
    )
    getattr(st, status_kind)(summary_md)

    with st.container(border=True):
        st.markdown(
            f"#### Annual Cohort Snapshot: Strict 3.50 Rule vs. {renewal:.2f} GPA Alternative"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Expected Returning Students",
            f"{a['expected_retained']:,.1f}",
            f"{delta['delta_expected_retained']:+,.1f} vs. strict 3.50",
            help="Sum of individual return probabilities across the cohort.",
        )
        c2.metric(
            "Expected Return Rate",
            fmt_pct(a["expected_retention_rate"]),
            f"{100.0 * delta['delta_expected_retention_rate']:+.1f} percentage pts",
            help="Average probability of reenrollment next year.",
        )
        c3.metric(
            "Scholarship + Tutoring Outlay",
            fmt_usd(a["total_institutional_spend"]),
            f"{fmt_signed_usd(delta['delta_total_institutional_spend'])} vs. strict 3.50",
            delta_color="inverse",
            help="Expected scholarship discounts for returning students plus direct tutoring costs.",
        )
        c4.metric(
            "Net Tuition Contribution",
            fmt_usd(a["expected_net_contribution"]),
            f"{fmt_signed_usd(delta['delta_expected_net_contribution'])} vs. strict 3.50",
            help="Gross tuition minus instructional costs, scholarship discounts, and tutoring spend.",
        )

    col_chart, col_text = st.columns([1.25, 1.0], gap="large")
    with col_chart:
        with st.container(border=True):
            st.plotly_chart(
                chart_gpa_distribution(cohort, alternative_policy),
                use_container_width=True,
            )
            st.caption(
                "Bars show pre-tutoring first-year college GPAs color-coded by award status under your alternative rule. "
                "The dashed crimson line marks the strict 3.50 cutoff."
            )
    with col_text:
        with st.container(border=True):
            st.markdown("#### What Changes at the GPA Cutoff?")
            st.markdown(explain_eligibility_shifts(b, a, alternative_policy))

    with st.container(border=True):
        st.markdown("#### Impact on Student Work Hours, Borrowing & Unpaid Gaps")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric(
            "Students Losing Any Aid",
            f"{a['affected_any_loss_count']:,}",
            f"{a['affected_any_loss_count'] - b['affected_any_loss_count']:+,} vs. strict 3.50",
            delta_color="inverse",
        )
        f2.metric(
            "Avg Extra Work (Affected)",
            f"{a['mean_work_hours_per_affected_student']:.1f} hrs/wk",
            help="Average weekly hours needed at the assumed wage among students who lose part or all of their scholarship.",
        )
        f3.metric(
            "Potential Extra Loans",
            fmt_usd(a["potential_extra_borrowing_total"]),
            f"{fmt_signed_usd(delta['delta_potential_extra_borrowing_total'])} vs. strict 3.50",
            delta_color="inverse",
        )
        f4.metric(
            "Unpaid Cash Shortfall",
            fmt_usd(a["potential_uncovered_gap_total"]),
            f"{fmt_signed_usd(delta['delta_potential_uncovered_gap_total'])} vs. strict 3.50",
            delta_color="inverse",
        )

        if financing.enabled:
            st.divider()
            j1, j2, j3 = st.columns(3)
            j1.metric(
                "Capital-Constrained Recipients",
                f"{a['capital_constrained_count']:,}",
                f"{a['poverty_proxy_count']:,} at poverty proxy",
                delta_color="off",
            )
            j2.metric(
                "Students Over Absorbable Gap",
                f"{a['students_over_absorbable_gap']:,}",
                f"{int(delta['delta_students_over_absorbable_gap']):+,} vs. strict 3.50",
                delta_color="inverse",
            )
            j3.metric(
                "Total Unabsorbable Gap",
                fmt_usd(a["excess_gap_total"]),
                f"{fmt_signed_usd(delta['delta_excess_gap_total'])} vs. strict 3.50",
                delta_color="inverse",
            )

        st.plotly_chart(chart_financing_breakdown(b, a), use_container_width=True)
        st.markdown(explain_student_finances(b, a, economics, financing))

    with st.container(border=True):
        st.markdown("#### Side-by-Side Institutional & Student Accounting Table")
        st.markdown(explain_budget_ledger(a, economics))
        ledger_rows = [
            ["Students keeping 100% scholarship", f"{b['full_renewal_count']:,}", f"{a['full_renewal_count']:,}", f"{a['full_renewal_count'] - b['full_renewal_count']:+,}"],
            ["Students on partial probation award", f"{b['partial_renewal_count']:,}", f"{a['partial_renewal_count']:,}", f"{a['partial_renewal_count'] - b['partial_renewal_count']:+,}"],
            ["Students losing 100% of scholarship", f"{b['full_loss_count']:,}", f"{a['full_loss_count']:,}", f"{a['full_loss_count'] - b['full_loss_count']:+,}"],
            ["Expected returning students", f"{b['expected_retained']:,.1f}", f"{a['expected_retained']:,.1f}", f"{delta['delta_expected_retained']:+,.1f}"],
            ["Multi-year graduation proxy (students)", f"{b['expected_completion_proxy']:,.1f}", f"{a['expected_completion_proxy']:,.1f}", f"{delta['delta_expected_completion_proxy']:+,.1f}"],
            ["Expected gross tuition revenue", fmt_usd(b["expected_gross_tuition"]), fmt_usd(a["expected_gross_tuition"]), fmt_signed_usd(delta["delta_expected_gross_tuition"])],
            ["Expected instructional cost", fmt_usd(b["expected_marginal_instruction_cost"]), fmt_usd(a["expected_marginal_instruction_cost"]), fmt_signed_usd(delta["delta_expected_marginal_instruction_cost"])],
            ["Expected scholarship + tutoring spend", fmt_usd(b["total_institutional_spend"]), fmt_usd(a["total_institutional_spend"]), fmt_signed_usd(delta["delta_total_institutional_spend"])],
            ["Net tuition contribution to university", fmt_usd(b["expected_net_contribution"]), fmt_usd(a["expected_net_contribution"]), fmt_signed_usd(delta["delta_expected_net_contribution"])],
            ["Potential extra student loans", fmt_usd(b["potential_extra_borrowing_total"]), fmt_usd(a["potential_extra_borrowing_total"]), fmt_signed_usd(delta["delta_potential_extra_borrowing_total"])],
            ["Potential unpaid cash shortfall", fmt_usd(b["potential_uncovered_gap_total"]), fmt_usd(a["potential_uncovered_gap_total"]), fmt_signed_usd(delta["delta_potential_uncovered_gap_total"])],
        ]
        if financing.enabled:
            ledger_rows.extend(
                [
                    ["Students exceeding absorbable gap", f"{b['students_over_absorbable_gap']:,}", f"{a['students_over_absorbable_gap']:,}", f"{int(delta['delta_students_over_absorbable_gap']):+,}"],
                    ["Total unabsorbable gap ($)", fmt_usd(b["excess_gap_total"]), fmt_usd(a["excess_gap_total"]), fmt_signed_usd(delta["delta_excess_gap_total"])],
                ]
            )
        summary_table = pd.DataFrame(
            ledger_rows,
            columns=["Outcome Measure", "Strict 3.50 Rule", "Proposed Alternative", "Difference"],
        )
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2 · BUDGET OPTIMIZER
# ===========================================================================
with tab_opt:
    with st.container(border=True):
        st.markdown("#### Find the Best Renewal Policy Within a Fixed Aid Budget")
        st.markdown(
            "Set the maximum **added scholarship + tutoring dollars** the university can spend above the strict 3.50 rule. "
            "The solver evaluates every GPA cutoff from **2.50 to 3.50** (in 0.05 steps), probation award shares (**0%, 25%, 50%, 75%, 100%**), "
            "and tutoring (**On vs. Off**), then selects the top-performing option within your budget ceiling."
        )
        ctrl1, ctrl2 = st.columns([1.2, 1.0], gap="large")
        with ctrl1:
            goal = st.radio(
                "Optimization Goal",
                [
                    "Returning students",
                    "Multi-year graduation proxy",
                    "Net tuition contribution",
                ],
                horizontal=True,
            )
        with ctrl2:
            budget = st.slider(
                "Max added scholarship + tutoring spend ($)",
                0,
                2_000_000,
                750_000,
                step=25_000,
            )

    goal_map = {
        "Returning students": (
            "additional_expected_retained",
            "Added Returning Students vs. Strict 3.50",
        ),
        "Multi-year graduation proxy": (
            "additional_completion_proxy",
            "Added Graduation-Proxy Outcomes vs. Strict 3.50",
        ),
        "Net tuition contribution": (
            "additional_net_contribution",
            "Change in Net Tuition Contribution ($)",
        ),
    }
    obj_key, obj_label = goal_map[goal]

    fp = hashlib.sha256(
        json.dumps({"b": b, "fc": asdict(financing)}, sort_keys=True).encode()
    ).hexdigest()
    candidates, _ = cached_optimization(
        cohort,
        economics,
        fp,
        probation_band,
        support_lift,
        support_band,
        support_cost,
        float(budget),
        financing,
    )
    feasible_df = candidates[candidates.additional_spend <= budget + 0.01]
    best = feasible_df.sort_values(
        [obj_key, "additional_expected_retained", "additional_spend"],
        ascending=[False, False, True],
    ).iloc[0]
    tutor_label = "Included" if best.pre_renewal_support else "Not included"

    with st.container(border=True):
        st.markdown(f"#### Recommended Policy — Maximizing {goal}")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Optimal Full-Renewal Cutoff", f"{best.renewal_gpa:.2f} GPA")
        o2.metric("Probation Award Share", f"{best.probation_award_pct:.0f}%", f"Tutoring: {tutor_label}")
        o3.metric("Added Returning Students", f"{best.additional_expected_retained:+.1f}")
        o4.metric("Added Aid + Tutoring Spend", fmt_usd(best.additional_spend))

        st.plotly_chart(
            chart_optimization_frontier(
                candidates, best, float(budget), obj_key, obj_label
            ),
            use_container_width=True,
        )

        opt_bullets = [
            f"- **Feasible candidate set:** **{len(feasible_df):,} of {len(candidates):,}** tested policy combinations stay at or below your **{md_usd(budget)}** incremental spending cap.",
            f"- **Top-scoring rule (gold star):** Renew 100% scholarships at **{best.renewal_gpa:.2f} GPA**, grant **{best.probation_award_pct:.0f}%** partial aid across the **{probation_band:.2f}-point** probation window, with early-warning tutoring **{tutor_label.lower()}**.",
        ]
        if best.additional_expected_retained > 0.05 and best.additional_spend > 0:
            opt_bullets.append(
                f"- **Gross outlay per added returner:** **{md_usd(best.additional_spend / best.additional_expected_retained)}** in incremental aid and tutoring spend per additional returning student."
            )
        opt_bullets.append(
            f"- **Net operating effect:** After crediting tuition paid by retained students, annual net tuition contribution shifts by **{md_signed_usd(best.additional_net_contribution)}**."
        )
        st.markdown("\n".join(opt_bullets))

    with st.container(border=True):
        st.markdown("#### Top 10 Affordable Policy Configurations")
        top10 = (
            feasible_df.sort_values(
                [obj_key, "additional_expected_retained", "additional_spend"],
                ascending=[False, False, True],
            )
            .head(10)
            .copy()
        )
        top10_display = pd.DataFrame(
            {
                "Full-Renewal GPA": top10["renewal_gpa"].map("{:.2f}".format),
                "Probation Share": top10["probation_award_pct"].map("{:.0f}%".format),
                "Tutoring": top10["pre_renewal_support"].map(lambda x: "Yes" if x else "No"),
                "Tutored Students": top10["support_students"],
                "Added Returners": top10["additional_expected_retained"].map("{:+.1f}".format),
                "Added Graduation Proxy": top10["additional_completion_proxy"].map("{:+.1f}".format),
                "Added Aid Spend": top10["additional_spend"].map(fmt_usd),
                "Net Tuition Margin Change": top10["additional_net_contribution"].map(fmt_signed_usd),
            }
        )
        st.dataframe(top10_display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download All Tested Policy Combinations (CSV)",
            candidates.to_csv(index=False),
            "scholarship_policy_frontier.csv",
            "text/csv",
        )

# ===========================================================================
# TAB 3 · SENSITIVITY & EQUITY
# ===========================================================================
with tab_sens:
    with st.container(border=True):
        st.markdown("#### 1. Which Behavioral or Financing Assumption Drives the Result Most?")
        st.markdown(
            "Because behavioral and liquidity inputs are assumed rather than estimated from campus trials, "
            "the tornado chart below perturbs **one parameter at a time** while holding all other settings fixed. "
            "Toggle **Poverty & Capitalization Constraints** in Sidebar Section 3 to include credit-access, work-cap, and absorbable-gap bars."
        )
        sensitivity = assumption_sensitivity(
            cohort, alternative_policy, economics, financing
        )
        st.plotly_chart(chart_tornado(sensitivity), use_container_width=True)

        widest = sensitivity.iloc[-1]
        min_swing = float(sensitivity[["low_delta", "high_delta"]].min().min())
        max_swing = float(sensitivity[["low_delta", "high_delta"]].max().max())
        if min_swing < 0 < max_swing:
            st.warning(
                f"**Decision Reversal Risk:** Depending on **{widest['assumption'].lower()}**, "
                f"the enrollment change crosses zero—ranging from **{min_swing:+.1f}** to **{max_swing:+.1f}** students. "
                "Estimate this parameter empirically before adopting a permanent rule change."
            )
        else:
            st.info(
                f"**Primary Sensitivity Driver — {widest['assumption']}:** Varying this input moves expected returning enrollment "
                f"between **{min(widest.low_delta, widest.high_delta):+.1f}** and **{max(widest.low_delta, widest.high_delta):+.1f}** students "
                f"(a **{widest.swing:.1f}-student** range). Although the sign stays consistent across bounds, "
                "this is the highest-priority parameter for Institutional Research to measure."
            )

    with st.container(border=True):
        st.markdown("#### 2. Outcomes Across Every Full-Renewal GPA Cutoff (2.50 to 3.80)")
        sweep = cached_sweep(
            cohort,
            economics,
            probation_band,
            probation_fraction,
            support_band,
            support_lift,
            support_cost,
            financing,
        )
        st.plotly_chart(chart_threshold_response(sweep, renewal), use_container_width=True)
        best_margin_row = sweep.loc[sweep.expected_net_contribution.idxmax()]
        st.markdown(
            f"- **Expected returning students (teal solid curve, left axis):** Increases steadily as the full-renewal GPA bar is lowered.\n"
            f"- **Net tuition contribution (amber dotted curve, right axis):** Peaks near a **{best_margin_row.threshold:.2f} GPA** cutoff at **{md_usd(best_margin_row.expected_net_contribution)}**, marking the point where incremental tuition from retained students balances rising scholarship renewals."
        )

    col_seed, col_eq = st.columns([0.95, 1.45], gap="large")
    with col_seed:
        with st.container(border=True):
            st.markdown("#### 3. Synthetic Cohort Resampling Check")
            if mode == "Synthetic cohort" and cohort_settings is not None:
                n_seeds = st.slider("Number of independent random seeds", 5, 40, 15, step=5)
                seeds = cached_seeds(
                    cohort_settings, alternative_policy, economics, n_seeds, financing
                )
                s1, s2 = st.columns(2)
                s1.metric("Mean Retention Gain", f"{seeds.delta_expected_retained.mean():+.1f} students")
                s2.metric(
                    "Min / Max Across Seeds",
                    f"{seeds.delta_expected_retained.min():+.1f} to {seeds.delta_expected_retained.max():+.1f}",
                )
                st.caption(
                    f"Re-simulating {n_seeds} independent cohorts confirms whether policy gains are stable across random draws."
                )
            else:
                st.info(
                    "Random-seed checks apply only to synthetic cohorts. Uploaded CSV files represent a fixed dataset."
                )

    with col_eq:
        with st.container(border=True):
            st.markdown("#### 4. Distributional Incidence Across Need, Poverty & Capital Access")
            groups = subgroup_summary(cohort, baseline, alternative, min_cell=15)
            if groups.empty:
                st.info("Subgroups have fewer than 15 students and are suppressed to protect privacy.")
            else:
                eq_df = pd.DataFrame(
                    {
                        "Dimension": groups["grouping"],
                        "Student Group": groups["group"],
                        "Students": groups["students"],
                        "Strict 3.50 Return": (100 * groups["strict_retention_rate"]).map("{:.1f}%".format),
                        "Alt Return": (100 * groups["alt_retention_rate"]).map("{:.1f}%".format),
                        "Gain (pp)": groups["retention_change_pp"].map("{:+.1f}".format),
                        "Avg Added Loans": groups["mean_potential_added_loans"].map(fmt_usd),
                        "Avg Unpaid Gap (Alt)": groups["mean_unfunded_gap"].map(fmt_usd),
                    }
                )
                st.dataframe(eq_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Groups reflect modeled financial attributes (award reliance, need-index tertiles, Pell-like poverty proxy, and capital access)—never protected classes. Cells under 15 students are suppressed."
                )

# ===========================================================================
# TAB 4 · GRADING & BENCHMARKS
# ===========================================================================
with tab_meas:
    with st.container(border=True):
        st.markdown("#### Distinguishing Readiness, Selection Effects & Section Grading Noise")
        gpa_diff = cohort.hs_gpa.mean() - cohort.college_gpa.mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Entry High-School GPA", f"{cohort.hs_gpa.mean():.2f}")
        m2.metric(
            "Mean First-Year College GPA",
            f"{cohort.college_gpa.mean():.2f}",
            f"{-gpa_diff:+.2f} vs. HS GPA",
        )
        if mode == "Synthetic cohort":
            mismatch = ((cohort.college_gpa >= 3.50) != (cohort.mastery_gpa_proxy >= 3.50)).mean()
            m3.metric(
                "3.50 Cutoff Classification Noise",
                fmt_pct(mismatch),
                help="Share of synthetic students whose observed college GPA lands on the opposite side of 3.50 from their latent mastery score due to section grading dispersion.",
            )
        else:
            m3.metric("Records Analyzed", f"{len(cohort):,}")

        st.markdown(
            f"- **Observed GPA shift:** Entering awardees average a **{cohort.hs_gpa.mean():.2f}** high-school GPA versus a **{cohort.college_gpa.mean():.2f}** first-year college GPA (a **{gpa_diff:+.2f}-point** difference).\n"
            "- **Why a GPA drop does not prove grade inflation:** Initial scholarship selection truncates the cohort at high incoming GPAs, first-year college courses introduce steeper grading scales, and grading leniency varies across course sections."
        )

        q_fig = chart_readiness_quintiles(cohort)
        if q_fig is not None:
            st.plotly_chart(q_fig, use_container_width=True)
        else:
            st.info("Include an optional `readiness_score` (0–100) column in your CSV to view GPA patterns across readiness quintiles.")

    col_sec, col_obs = st.columns(2, gap="large")
    with col_sec:
        with st.container(border=True):
            st.markdown("#### Anonymous Course-Section Grade Residuals")
            sec_df = section_grade_diagnostic(cohort, min_cell=15)
            if sec_df.empty:
                st.info("No anonymous course sections contain at least 15 students.")
            else:
                sec_view = pd.DataFrame(
                    {
                        "Section Code": sec_df["section"],
                        "Students": sec_df["students"],
                        "Mean College GPA": sec_df["average_college_gpa"].map("{:.2f}".format),
                        "Adjusted Residual": sec_df["mean_grade_residual"].map("{:+.2f}".format),
                    }
                )
                st.dataframe(sec_view, use_container_width=True, hide_index=True, height=280)
                st.caption(
                    "Residuals measure how far a section's average GPA sits above or below regression predictions given incoming HS GPA and readiness score. Section differences can reflect course rigor or major mix—not instructor misconduct."
                )

    with col_obs:
        with st.container(border=True):
            st.markdown("#### Observed Near-Cutoff Persistence (Uploaded CSV Only)")
            if mode == "Approved de-identified CSV" and "reenrolled_next_term" in cohort:
                diag = near_cutoff_diagnostic(cohort, cutoff=3.50, bandwidth=0.20)
                st.metric(
                    "Overall Observed Return Rate",
                    fmt_pct(float(cohort.reenrolled_next_term.mean())),
                )
                if diag:
                    d1, d2 = st.columns(2)
                    d1.metric(
                        "Return Rate (3.30–3.49 GPA)",
                        fmt_pct(diag["retention_below"]),
                        f"n = {diag['n_below']}",
                    )
                    d2.metric(
                        "Return Rate (3.50–3.70 GPA)",
                        fmt_pct(diag["retention_above"]),
                        f"n = {diag['n_above']}",
                    )
                    st.caption(
                        f"Students just above 3.50 persisted at a rate **{100*diag['difference_above_minus_below']:+.1f} pp** different from those just below. "
                        "Descriptive difference only—causal regression discontinuity requires testing for GPA bunching and appeals."
                    )
                else:
                    st.info("Fewer than 10 students appear within ±0.20 GPA of 3.50.")
            elif mode == "Approved de-identified CSV":
                st.info("Add an optional 0/1 `reenrolled_next_term` column to display observed near-cutoff return rates.")
            else:
                st.info(
                    "Synthetic mode omits fabricated 'observed' retention flags so scenario projections are never mistaken for empirical campus data."
                )

    with st.container(border=True):
        st.markdown("#### Optional Federal & Macroeconomic Context Benchmarks")
        st.caption(
            "Query live institutional context from the U.S. College Scorecard or national labor indicators from FRED. Public benchmarks do not alter internal simulation parameters."
        )
        b_col1, b_col2 = st.columns(2, gap="large")

        scorecard_key = key_from_env_or_secrets("COLLEGE_SCORECARD_API_KEY") or key_from_env_or_secrets(
            "DATA_GOV_API_KEY"
        )
        with b_col1:
            st.markdown("**U.S. College Scorecard Lookup**")
            school_name = st.text_input("Institution Name", value="Cheyney University", max_chars=85)
            if st.button("Fetch College Scorecard Data", disabled=not scorecard_key):
                try:
                    fields = [
                        "id",
                        "school.name",
                        "school.state",
                        "latest.student.size",
                        "latest.aid.pell_grant_rate",
                        "latest.student.retention_rate.four_year.full_time",
                        "latest.cost.tuition.in_state",
                    ]
                    resp = requests.get(
                        "https://api.data.gov/ed/collegescorecard/v1/schools",
                        params={
                            "api_key": scorecard_key,
                            "school.name": school_name.strip(),
                            "fields": ",".join(fields),
                            "per_page": 8,
                        },
                        timeout=18,
                    )
                    resp.raise_for_status()
                    recs = resp.json().get("results", [])
                    rows = []
                    for r in recs:
                        pell = r.get("latest.aid.pell_grant_rate")
                        ret = r.get("latest.student.retention_rate.four_year.full_time")
                        tuit = r.get("latest.cost.tuition.in_state")
                        rows.append(
                            {
                                "School": r.get("school.name") or "N/A",
                                "State": r.get("school.state") or "N/A",
                                "Enrollment": r.get("latest.student.size") or "N/A",
                                "Pell Share": f"{100*float(pell):.1f}%" if isinstance(pell, (int, float)) else "N/A",
                                "FT Retention": f"{100*float(ret):.1f}%" if isinstance(ret, (int, float)) else "N/A",
                                "In-State Tuition": fmt_usd(float(tuit)) if isinstance(tuit, (int, float)) else "N/A",
                            }
                        )
                    st.session_state["scorecard_rows"] = rows
                    st.session_state["scorecard_query"] = school_name.strip()
                except Exception as exc:
                    st.error(f"Scorecard lookup failed ({type(exc).__name__}).")

            if st.session_state.get("scorecard_rows") and st.session_state.get("scorecard_query") == school_name.strip():
                st.dataframe(pd.DataFrame(st.session_state["scorecard_rows"]), use_container_width=True, hide_index=True)
            elif not scorecard_key:
                st.caption("Configure `COLLEGE_SCORECARD_API_KEY` in `.streamlit/secrets.toml` to enable.")

        fred_key = key_from_env_or_secrets("FRED_API_KEY")
        with b_col2:
            st.markdown("**Federal Reserve (FRED) Macro Context**")
            if st.button("Fetch U.S. Unemployment & CPI", disabled=not fred_key):
                try:
                    f_rows = []
                    for code, label in (("UNRATE", "U.S. Unemployment Rate (%)"), ("CPIAUCSL", "U.S. Consumer Price Index")):
                        resp = requests.get(
                            "https://api.stlouisfed.org/fred/series/observations",
                            params={
                                "series_id": code,
                                "api_key": fred_key,
                                "file_type": "json",
                                "sort_order": "desc",
                                "limit": 5,
                            },
                            timeout=18,
                        )
                        resp.raise_for_status()
                        obs = next(
                            (o for o in resp.json().get("observations", []) if o.get("value") not in (None, ".")),
                            None,
                        )
                        if obs:
                            f_rows.append({"Indicator": label, "Date": obs["date"], "Value": float(obs["value"])})
                    st.session_state["fred_rows"] = f_rows
                except Exception as exc:
                    st.error(f"FRED lookup failed ({type(exc).__name__}).")

            if st.session_state.get("fred_rows"):
                st.dataframe(pd.DataFrame(st.session_state["fred_rows"]), use_container_width=True, hide_index=True)
            elif not fred_key:
                st.caption("Configure `FRED_API_KEY` in `.streamlit/secrets.toml` to enable.")

# ===========================================================================
# TAB 5 · ADMINISTRATION BRIEF
# ===========================================================================
with tab_brief:
    memo_screen = build_policy_memo(
        mode,
        baseline,
        alternative,
        delta,
        alternative_policy,
        economics,
        cohort,
        financing,
        for_streamlit=True,
    )
    memo_download = build_policy_memo(
        mode,
        baseline,
        alternative,
        delta,
        alternative_policy,
        economics,
        cohort,
        financing,
        for_streamlit=False,
    )
    scenario_json = {
        "exported_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "app": "HBCU Scholarship Economics & Student Success Lab",
        "cohort_settings": asdict(cohort_settings) if cohort_settings is not None else "uploaded CSV",
        **aggregate_payload(
            mode, baseline, alternative, delta, alternative_policy, economics, financing
        ),
    }

    btn1, btn2 = st.columns(2)
    with btn1:
        st.download_button(
            "Download Administration Policy Brief (.md)",
            memo_download,
            "scholarship_administration_brief.md",
            "text/markdown",
            use_container_width=True,
        )
    with btn2:
        st.download_button(
            "Download Reproducible Scenario Audit (.json)",
            json.dumps(scenario_json, indent=2, sort_keys=True),
            "scholarship_scenario_audit.json",
            "application/json",
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown(memo_screen)

    with st.container(border=True):
        st.markdown("#### Optional AI Discussion Summary (Aggregate Metrics Only)")
        openai_key = key_from_env_or_secrets("OPENAI_API_KEY")
        outgoing = aggregate_payload(
            mode, baseline, alternative, delta, alternative_policy, economics, financing
        )
        out_fp = hashlib.sha256(json.dumps(outgoing, sort_keys=True).encode()).hexdigest()

        if not openai_key:
            st.caption(
                "Optional: configure `OPENAI_API_KEY` in `.streamlit/secrets.toml` to generate additional discussion notes. "
                "All calculations, charts, and the Administration Brief above run offline without any API key."
            )
        else:
            st.caption(
                "Privacy safeguard: clicking below transmits **only** aggregate cohort totals and slider assumptions—never student-level rows."
            )

        if st.button(
            "Generate Plain-English Discussion Notes (OpenAI)",
            disabled=(not openai_key or len(cohort) < 25),
        ):
            try:
                model_id = key_from_env_or_secrets("OPENAI_MODEL") or "gpt-4.1-mini"
                with st.spinner("Drafting summary from aggregate scenario totals..."):
                    st.session_state["ai_narrative"] = request_openai_interpretation(
                        openai_key, model_id, outgoing
                    )
                    st.session_state["ai_fingerprint"] = out_fp
            except Exception as exc:
                st.error(f"Could not generate OpenAI summary ({type(exc).__name__}).")

        if st.session_state.get("ai_narrative"):
            if st.session_state.get("ai_fingerprint") == out_fp:
                st.markdown(escape_dollar_for_streamlit(st.session_state["ai_narrative"]))
            else:
                st.info("Sidebar settings changed since this note was generated. Click above to refresh.")

# ===========================================================================
# TAB 6 · FORMULAS & LAB GUIDE
# ===========================================================================
with tab_guide:
    eq_left, eq_right = st.columns(2, gap="large")
    with eq_left:
        with st.container(border=True):
            st.markdown("#### 1. Scholarship Renewal Rule")
            st.latex(
                r"A_i = \begin{cases} S_i, & G_i^* \geq \tau \\ \alpha S_i, & \tau - b \leq G_i^* < \tau \\ 0, & G_i^* < \tau - b \end{cases}"
            )
            st.markdown(
                "- **$S_i$**: Initial annual scholarship award\n"
                "- **$G_i^*$**: Effective first-year GPA (including any pre-renewal tutoring lift)\n"
                "- **$\\tau$**: Full-renewal GPA cutoff\n"
                "- **$b$**: Probation window width below $\\tau$, retaining fraction $\\alpha$ of the award"
            )

            st.markdown("#### 2. Unconstrained Work & Borrowing Substitution")
            st.latex(
                r"\Delta H_i = \frac{(S_i - A_i)\,w_s}{w_h \cdot 32}, \qquad \Delta L_i = (S_i - A_i)\,\ell_s"
            )
            st.markdown(
                "- **$(S_i - A_i)$**: Lost scholarship funding\n"
                "- **$w_s, \\ell_s$**: Target shares of lost aid replaced by wages versus student loans\n"
                "- **$w_h$**: Assumed student hourly wage across a 32-week academic year"
            )

    with eq_right:
        with st.container(border=True):
            st.markdown("#### 3. Expected Return Probability")
            st.latex(
                r"p_i = \mathrm{clip}_{[0.02,\,0.99]}\!\left(p_i^0 + 0.16(G_i^* - G_i) - \beta\,\frac{S_i - A_i}{S_i}\,v_i - \text{strain}_i\right)"
            )
            st.markdown(
                "- **$p_i^0$**: Baseline persistence probability (adjusted for GPA and financial need)\n"
                "- **$\\beta$**: Assumed maximum retention drop from losing 100% of aid\n"
                "- **$v_i$**: Financial vulnerability weight; **$\\text{strain}_i$**: optional unabsorbable-gap kink"
            )

            st.markdown("#### 4. University Net Tuition Contribution")
            st.latex(
                r"\text{Net Contribution} = \sum_{i=1}^{N} p_i\left(\text{Tuition} - \text{Cost} - A_i\right) - C_{\text{tutoring}}"
            )
            st.markdown(
                "Measures whether incremental tuition revenue from retained students offsets the added cost of scholarship renewals and tutoring."
            )

    with st.container(border=True):
        st.markdown("#### 5. Poverty & Capitalization Constraints Module (Optional)")
        st.latex(
            r"\Delta H_i = \min\!\left(\frac{(S_i - A_i)\,w_s}{w_h T},\;\bar{H}\right), \qquad \Delta L_i = \min\!\big((S_i - A_i)\,\ell_s\,\kappa_i,\;\bar{L}\big)"
        )
        st.latex(
            r"\text{Gap}_i = (S_i - A_i) - w_h T\,\Delta H_i - \Delta L_i, \qquad \text{strain}_i = \gamma\,\min\!\left(1,\;\frac{\max(0,\,\text{Gap}_i - \bar{A}_i)}{S_i}\right)"
        )
        st.markdown(
            "- **Flow vs. Stock Distinction:** Poverty operates as a household **cash-flow** constraint (low absorbable buffer $\\bar{A}_i$), whereas capitalization operates as a **wealth/credit stock** constraint (reduced borrowing access $\\kappa_i$ and loan ceiling $\\bar{L}$). Two students with identical `need_index` scores can face sharply different persistence feasibility when an award is lost.\n"
            "- **Literature Motivation (Channels Only, Not Calibrated Campus Parameters):** Credit constraints and dropout (Stinebrickner & Stinebrickner, 2008; Lochner & Monge-Naranjo, 2011); term-time employment and academic performance (Stinebrickner & Stinebrickner, 2003); student loan limits and attainment (Marx & Turner, 2019; Denning & Jones, 2019); Pell grant liquidity and graduation (Denning, Marx & Turner, 2019); basic-needs insecurity at HBCUs (Hope Center)."
        )

    with st.container(border=True):
        st.markdown("#### Classroom Lab Exercises for Economics Students")
        st.markdown(
            "1. **Evaluate the 3.50 Cliff:** Set the alternative cutoff to **3.20** with **0%** probation and **no tutoring**, then add a **0.25-point** probation window at **50%** aid. Compare how many students forfeit 100% of their award at each step.\n"
            "2. **Test Flow vs. Stock Liquidity Limits:** Toggle on **Poverty & Capitalization Constraints** in Sidebar Section 3. How many students cross their absorbable unpaid gap under the strict 3.50 rule versus your alternative, and how does the Tornado chart in **Tab 3** change?\n"
            "3. **Locate the Break-Even Tuition Margin:** Increase *Instructional cost per returning student* from **\\$6,500** toward **\\$12,000**. Identify the point at which retaining additional students stops paying for itself in net university contribution.\n"
            "4. **Solve a Constrained Budget Optimization:** In **Tab 2**, cap added spending at **\\$250,000**. Does the optimizer prioritize lowering the universal GPA cutoff or funding a targeted probation band with tutoring?\n"
            "5. **Design an Empirical Campus Study:** Inspect **Tab 3** to identify the assumption with the widest tornado bar. Draft a brief empirical strategy explaining what longitudinal registrar and financial-aid records Institutional Research would need to estimate that parameter."
        )

    ref_left, ref_right = st.columns(2, gap="large")
    with ref_left:
        with st.container(border=True):
            st.markdown("#### Research & Policy References")
            for label, link in STUDY_LINKS.items():
                st.markdown(f"- [{label}]({link})")
            if mode == "Synthetic cohort":
                st.download_button(
                    "Download Synthetic Classroom Dataset (CSV)",
                    cohort.drop(columns=["mastery_gpa_proxy"], errors="ignore").to_csv(index=False),
                    "scholarship_synthetic_teaching_cohort.csv",
                    "text/csv",
                )

    with ref_right:
        with st.container(border=True):
            st.markdown("#### Governance & Privacy Guardrails")
            st.markdown(
                "- **No automated individual aid decisions:** This laboratory evaluates cohort-level policy scenarios, never individual student eligibility.\n"
                "- **Institutional merit vs. Federal Aid:** Campus scholarship rules are distinct from federal Pell Grants and Satisfactory Academic Progress (SAP).\n"
                "- **Poverty & capital-access flags are proxies:** Pell-like status, maxed federal loans, and PLUS denials mark financing exposure rather than directly measuring household resources; rank-imputed flags are exploratory stand-ins.\n"
                "- **FERPA compliance:** Upload only approved, de-identified datasets; subgroups below 15 students are automatically suppressed."
            )
