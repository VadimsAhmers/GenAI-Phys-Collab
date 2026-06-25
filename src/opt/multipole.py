"""Multipole channel powers and the channel-distribution loss.

Pure functions over the coefficient dict {"a"|"b": {n: {m: complex}}} -- no COMSOL,
no LLM. This is the live "formulation 2" objective agreed with the physicists:
steer scattered power into chosen multipole channels.

The loss mirrors new_conversation/multipole_loss.py (the spec from Vsevolod);
kept here as the single in-repo source so the optimizer does not depend on the
hand-off folder.
"""
from __future__ import annotations

# Channel labels by multipole order n (electric "a" / magnetic "b").
NAMES_E = {1: "ED", 2: "EQ", 3: "EO", 4: "EH"}
NAMES_M = {1: "MD", 2: "MQ", 3: "MO", 4: "MH"}

# Canonical channel order (electric orders then magnetic), for tables/plots.
CHANNEL_ORDER = list(NAMES_E.values()) + list(NAMES_M.values())


def validate_target(target: dict[str, float]) -> None:
    """Fail early (clear message) on a malformed channel-distribution target,
    instead of a deep KeyError/ZeroDivision inside the loss."""
    if not target:
        raise ValueError("target is empty; specify at least one channel share")
    unknown = set(target) - set(CHANNEL_ORDER)
    if unknown:
        raise ValueError(
            f"unknown channel(s) {sorted(unknown)}; valid: {CHANNEL_ORDER}"
        )
    total = sum(target.values())
    if total <= 0:
        raise ValueError(f"target shares must sum to a positive value, got {total}")


def channel_powers(coeffs: dict, M: int = 4) -> dict[str, float]:
    """Power per channel: P_En = sum_m |a_nm|^2, P_Mn = sum_m |b_nm|^2."""
    P: dict[str, float] = {}
    for n in range(1, M + 1):
        pE = sum(abs(coeffs["a"][n][m]) ** 2 for m in range(-n, n + 1))
        pM = sum(abs(coeffs["b"][n][m]) ** 2 for m in range(-n, n + 1))
        P[NAMES_E.get(n, f"E{n}")] = float(pE)
        P[NAMES_M.get(n, f"M{n}")] = float(pM)
    return P


def multipole_loss(
    coeffs: dict,
    target: dict[str, float],
    M: int = 4,
    beta: float = 0.0,
    P_min: float = 1e-8,
    eps: float = 1e-16,
) -> tuple[float, dict]:
    """Channel-distribution loss L = sum_i (q_i - t_i)^2 (+ optional beta penalty
    against the trivial zero-power solution).

    Args:
        coeffs: multipole coefficients {"a"/"b": {n: {m: complex}}}.
        target: desired channel shares, e.g. {"ED": 1.0} or {"ED": .5, "MD": .5}
            (need not be normalized; normalized internally).
        beta, P_min: weight and floor for the power-presence penalty.
    Returns:
        (loss, info) with info = {"P", "q", "P_total", "P_target"}.
    """
    P = channel_powers(coeffs, M)
    tgt = {k: v / sum(target.values()) for k, v in target.items()}

    P_tot = sum(P.values())
    q = {k: v / (P_tot + eps) for k, v in P.items()}

    L = sum((q[k] - tgt.get(k, 0.0)) ** 2 for k in P)

    P_target = sum(P[k] for k in tgt)
    L += beta * max(0.0, (P_min - P_target) / P_min) ** 2

    return float(L), {"P": P, "q": q, "P_total": P_tot, "P_target": P_target}


def feedback_string(info: dict, target: dict[str, float]) -> str:
    """Human/LLM-readable channel breakdown: shares q vs target, sorted by q."""
    tgt = {k: v / sum(target.values()) for k, v in target.items()}
    q = info["q"]
    rows = sorted(q.items(), key=lambda kv: kv[1], reverse=True)
    parts = []
    for k, v in rows:
        t = tgt.get(k, 0.0)
        marker = f" (target {t:.2f})" if k in tgt else ""
        parts.append(f"{k}={v:.3f}{marker}")
    return (
        "power shares q -> " + ", ".join(parts)
        + f"; P_total={info['P_total']:.3e}, P_in_target={info['P_target']:.3e}"
    )
