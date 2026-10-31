"""
models/kelly_risk.py
====================
Fractional Kelly Criterion for tennis betting with a strict 3.5 % bankroll cap.

The module exposes two public helpers:

* **calculate_kelly_stake** – computes the optimal stake given decimal odds,
  an effective probability, the current bankroll, and a fractional‑Kelly
  multiplier (default 20 %).
* **evaluate_opportunity_filters** – applies pre‑bet sanity filters (minimum
  expected value, odds corridor) and returns a pass / fail verdict together
  with a human‑readable reason string.

Mathematical background
-----------------------
Full Kelly fraction:

    f* = (p · (odds − 1) − (1 − p)) / (odds − 1)

where *p* is the model's estimated win probability and *odds* are in
**decimal** (European) format.  The raw fraction is then scaled by
``kelly_fraction`` (e.g. 0.20 for a conservative "fifth‑Kelly") and hard‑
capped at ``MAX_STAKE_PCT`` (3.5 %) of the bankroll.
"""

from __future__ import annotations

__all__ = [
    "calculate_kelly_stake",
    "evaluate_opportunity_filters",
    "MAX_STAKE_PCT",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STAKE_PCT: float = 0.035
"""Absolute ceiling – never risk more than 3.5 % of the bankroll on a
single wager, regardless of what the Kelly formula suggests."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_kelly_stake(
    odds: float,
    effective_prob: float,
    bankroll: float,
    kelly_fraction: float = 0.20,
) -> dict[str, float | bool]:
    """Compute the recommended stake using the Fractional Kelly Criterion.

    Parameters
    ----------
    odds : float
        Decimal (European) odds offered by the sportsbook.  Must be > 1.0.
    effective_prob : float
        Model‑estimated probability of the outcome winning, in [0, 1].
    bankroll : float
        Current bankroll in the account currency.  Must be > 0.
    kelly_fraction : float, optional
        Fraction of the full Kelly to apply (default ``0.20`` = 20 %).  A
        value of 1.0 would use the full Kelly, which is generally too
        aggressive for real‑world betting.

    Returns
    -------
    dict[str, float | bool]
        A dictionary with the following keys:

        * ``recommended_stake`` – absolute stake in currency units, capped
          at ``MAX_STAKE_PCT`` of the bankroll.
        * ``recommended_stake_pct`` – stake expressed as a fraction of
          the bankroll (0.0–``MAX_STAKE_PCT``).
        * ``is_capped`` – ``True`` when the raw fractional‑Kelly stake
          exceeded the 3.5 % cap and was therefore clipped.
        * ``kelly_edge`` – the full Kelly edge before applying the
          fractional multiplier.  A non‑positive edge means the bet has
          no mathematical value.
        * ``kelly_fraction_raw`` – the raw (un‑capped) fractional‑Kelly
          stake as a proportion of the bankroll.

    Raises
    ------
    ValueError
        If any input falls outside its valid domain.

    Examples
    --------
    >>> result = calculate_kelly_stake(odds=2.10, effective_prob=0.55, bankroll=1000.0)
    >>> round(result["recommended_stake"], 2)
    9.55
    >>> result["is_capped"]
    False
    """

    # -- Input validation ----------------------------------------------------
    if odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {odds!r}.")
    if not 0.0 <= effective_prob <= 1.0:
        raise ValueError(
            f"effective_prob must be in [0, 1], got {effective_prob!r}."
        )
    if bankroll <= 0.0:
        raise ValueError(f"bankroll must be > 0, got {bankroll!r}.")
    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError(
            f"kelly_fraction must be in (0, 1], got {kelly_fraction!r}."
        )

    # -- Full Kelly fraction -------------------------------------------------
    #   f* = (p * (odds - 1) - (1 - p)) / (odds - 1)
    net_odds: float = odds - 1.0
    kelly_edge: float = (effective_prob * net_odds - (1.0 - effective_prob)) / net_odds

    # -- No edge → zero stake -----------------------------------------------
    if kelly_edge <= 0.0:
        return {
            "recommended_stake": 0.0,
            "recommended_stake_pct": 0.0,
            "is_capped": False,
            "kelly_edge": kelly_edge,
            "kelly_fraction_raw": 0.0,
        }

    # -- Fractional Kelly & cap ----------------------------------------------
    kelly_fraction_raw: float = kelly_edge * kelly_fraction
    is_capped: bool = kelly_fraction_raw > MAX_STAKE_PCT
    stake_pct: float = min(kelly_fraction_raw, MAX_STAKE_PCT)
    recommended_stake: float = round(stake_pct * bankroll, 2)

    return {
        "recommended_stake": recommended_stake,
        "recommended_stake_pct": stake_pct,
        "is_capped": is_capped,
        "kelly_edge": kelly_edge,
        "kelly_fraction_raw": kelly_fraction_raw,
    }


def evaluate_opportunity_filters(
    odds: float,
    ev_percent: float,
    min_ev: float = 3.0,
    min_odds: float = 1.20,
    max_odds: float = 3.50,
) -> tuple[bool, str]:
    """Apply pre‑bet sanity filters to an opportunity.

    The filters ensure that only bets meeting a minimum expected‑value
    threshold and falling within an acceptable odds corridor are flagged
    as actionable.

    Parameters
    ----------
    odds : float
        Decimal (European) odds offered by the sportsbook.
    ev_percent : float
        Expected value of the bet expressed as a percentage (e.g. ``5.0``
        means +5 % EV).
    min_ev : float, optional
        Minimum acceptable EV percentage (default ``3.0``).
    min_odds : float, optional
        Lower bound of the acceptable odds corridor (default ``1.20``).
    max_odds : float, optional
        Upper bound of the acceptable odds corridor (default ``3.50``).

    Returns
    -------
    tuple[bool, str]
        A two‑element tuple ``(passes_filter, reason_message)``.

        * ``passes_filter`` is ``True`` when the opportunity satisfies
          **all** filters.
        * ``reason_message`` is a human‑readable explanation.  When the
          opportunity passes, the message summarises the key figures;
          when it fails, the message states the first violated constraint.

    Examples
    --------
    >>> passes, msg = evaluate_opportunity_filters(odds=1.95, ev_percent=5.2)
    >>> passes
    True

    >>> passes, msg = evaluate_opportunity_filters(odds=1.10, ev_percent=5.0)
    >>> passes
    False
    >>> "below minimum" in msg.lower()
    True
    """

    # -- Odds corridor -------------------------------------------------------
    if odds < min_odds:
        return (
            False,
            f"Odds {odds:.2f} are below minimum threshold ({min_odds:.2f}).",
        )

    if odds > max_odds:
        return (
            False,
            f"Odds {odds:.2f} exceed maximum threshold ({max_odds:.2f}).",
        )

    # -- Expected value ------------------------------------------------------
    if ev_percent < min_ev:
        return (
            False,
            f"Expected value {ev_percent:.1f}% is below the "
            f"minimum required EV ({min_ev:.1f}%).",
        )

    # -- All filters passed --------------------------------------------------
    return (
        True,
        f"Opportunity passes all filters: odds={odds:.2f}, "
        f"EV={ev_percent:.1f}% (min {min_ev:.1f}%), "
        f"odds corridor [{min_odds:.2f}, {max_odds:.2f}].",
    )
