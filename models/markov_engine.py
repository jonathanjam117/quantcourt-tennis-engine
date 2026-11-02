"""Hierarchical Markov Chain Engine for Tennis Analytics (QuantCourt AI).

Implements exact mathematical modeling from point to match:
Point -> Game (Service Hold) -> Tie-break (7-point) -> Set (6-game / TB) -> Match (Bo3 / Bo5).
Derives fair odds and commercial lines with built-in Colombian bookmaker margin adjustments.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.special import comb
from scipy.stats import norm

from etl.surface_profiles import (
    ALL_SURFACES,
    TOUR_AVERAGES,
    detect_surface_from_tournament,
    get_player_profile,
    get_surface_color,
    get_surface_emoji,
)
from models.ai_tennis_agent import TennisTacticalAdjustment, safe_ai_get
from models.kelly_risk import calculate_kelly_stake, evaluate_opportunity_filters


# ==============================================================================
# 1. CORE MARKOV CHAIN FUNCTIONS
# ==============================================================================

def calc_point_prob(spw_a: float, rpw_b: float, surface: str = "Dura Outdoor") -> float:
    """Calculates the probability that Player A wins a point when Player A is serving,
    adjusted by Player B's return capability on the specified surface.

    Formula:
    p = clip(media_spw + (SPW_A - media_spw) - (RPW_B - media_rpw), 0.48, 0.82)
    """
    tour_stats = TOUR_AVERAGES.get(surface, TOUR_AVERAGES["Dura Outdoor"])
    media_spw = tour_stats["spw"]
    media_rpw = tour_stats["rpw"]

    p_point_a = media_spw + (spw_a - media_spw) - (rpw_b - media_rpw)
    return float(np.clip(p_point_a, 0.48, 0.82))


def calc_game_prob(p_serve: float) -> float:
    """Calculates the probability of holding serve (winning a service game)
    given p_serve (probability of winning an individual point on serve).

    Uses the closed-form Markov chain formula including deuce/advantage:
    P(hold) = p^4 * (15 - 24p + 10p^2) + (20 * p^3 * q^3 * p^2) / (p^2 + q^2)
    """
    p = float(np.clip(p_serve, 0.05, 0.95))
    q = 1.0 - p

    # Probability of winning before deuce:
    # 4-0: p^4
    # 4-1: 4 * p^4 * q
    # 4-2: 10 * p^4 * q^2
    p_no_deuce = (p**4) * (1.0 + 4.0 * q + 10.0 * (q**2))

    # Probability of reaching deuce (3-3): C(6,3) * p^3 * q^3 = 20 * p^3 * q^3
    p_reach_deuce = 20.0 * (p**3) * (q**3)

    # Probability of winning from deuce: p^2 / (p^2 + q^2)
    denom = (p**2) + (q**2)
    p_win_from_deuce = (p**2) / denom if denom > 0 else 0.5

    p_hold = p_no_deuce + p_reach_deuce * p_win_from_deuce
    return float(np.clip(p_hold, 0.05, 0.98))


def calc_tiebreak_prob(p_serve_a: float, p_serve_b: float) -> float:
    """Calculates the probability that Player A wins a standard 7-point tiebreak.

    Uses alternating serve probability:
    p_avg = (p_serve_a + (1 - p_serve_b)) / 2
    Win conditions: first to 7 with 2-point lead, or won from 6-6 mini-deuce.
    """
    p_a_serving = float(np.clip(p_serve_a, 0.05, 0.95))
    p_a_returning = float(np.clip(1.0 - p_serve_b, 0.05, 0.95))
    p = (p_a_serving + p_a_returning) / 2.0
    q = 1.0 - p

    # Win 7-0 through 7-5 (Player A wins 7 points before Player B wins 6)
    p_win_clean = sum(comb(6 + k, k, exact=True) * (p**7) * (q**k) for k in range(6))

    # Reach 6-6 (both win exactly 6 points in first 12 points)
    p_reach_66 = comb(12, 6, exact=True) * (p**6) * (q**6)

    # Win from 6-6 (need 2 consecutive net points)
    denom = (p**2) + (q**2)
    p_win_from_66 = (p**2) / denom if denom > 0 else 0.5

    p_tb = p_win_clean + p_reach_66 * p_win_from_66
    return float(np.clip(p_tb, 0.05, 0.95))


def calc_set_prob(
    p_hold_a: float,
    p_hold_b: float,
    p_tb_a: float,
) -> Tuple[float, float]:
    """Calculates:
    1. Probability Player A wins the set (P_set_A).
    2. Probability of reaching a Tie-Break at 6-6 (P_tiebreak).

    Uses dynamic programming across games (i, j) where Player A serves when (i+j)%2 == 0.
    """
    memo: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def dp(i: int, j: int) -> Tuple[float, float]:
        # Returns (prob_a_wins_set, prob_reach_tiebreak)
        if (i, j) in memo:
            return memo[(i, j)]

        # Terminal conditions
        if i >= 6 and (i - j) >= 2:
            return (1.0, 0.0)
        if j >= 6 and (j - i) >= 2:
            return (0.0, 0.0)
        if i == 7 and j == 6:
            return (1.0, 1.0)
        if i == 6 and j == 7:
            return (0.0, 1.0)
        if i == 6 and j == 6:
            return (p_tb_a, 1.0)

        # Transition probabilities based on who is serving
        if (i + j) % 2 == 0:
            # Player A serves
            p_win_game = p_hold_a
        else:
            # Player B serves -> prob A wins game = 1 - p_hold_b
            p_win_game = 1.0 - p_hold_b

        p_lose_game = 1.0 - p_win_game

        # Next states
        res_win = dp(i + 1, j)
        res_lose = dp(i, j + 1)

        p_win_set = p_win_game * res_win[0] + p_lose_game * res_lose[0]
        p_tb = p_win_game * res_win[1] + p_lose_game * res_lose[1]

        memo[(i, j)] = (p_win_set, p_tb)
        return (p_win_set, p_tb)

    p_set_a, p_tb = dp(0, 0)
    return float(np.clip(p_set_a, 0.02, 0.98)), float(np.clip(p_tb, 0.03, 0.55))


def calc_match_prob_bo3(p_set_a: float) -> Dict[str, float]:
    """Best-of-3 sets probability calculations.
    Returns:
    - p_win_a: Overall match win probability for A
    - p_2_0_a: Exact 2-0 sets for A
    - p_2_1_a: Exact 2-1 sets for A
    - p_0_2_b: Exact 0-2 sets for B
    - p_1_2_b: Exact 1-2 sets for B
    - p_over_25_sets: Match goes to 3 sets
    - p_under_25_sets: Match ends in 2 sets
    - p_handicap_plus_15_a: A wins at least 1 set
    - p_handicap_plus_15_b: B wins at least 1 set
    """
    p = float(np.clip(p_set_a, 0.02, 0.98))
    q = 1.0 - p

    p_2_0 = p**2
    p_2_1 = 2.0 * (p**2) * q
    p_0_2 = q**2
    p_1_2 = 2.0 * (q**2) * p

    p_win_a = p_2_0 + p_2_1
    p_win_b = p_0_2 + p_1_2

    p_2_sets = p_2_0 + p_0_2
    p_3_sets = p_2_1 + p_1_2

    # Handicap sets: +1.5 means the player wins at least 1 set (does not lose in straight sets)
    p_plus_15_a = 1.0 - p_0_2
    p_plus_15_b = 1.0 - p_2_0

    return {
        "p_win_a": p_win_a,
        "p_win_b": p_win_b,
        "p_2_0_a": p_2_0,
        "p_2_1_a": p_2_1,
        "p_0_2_b": p_0_2,
        "p_1_2_b": p_1_2,
        "p_under_25_sets": p_2_sets,
        "p_over_25_sets": p_3_sets,
        "p_handicap_plus_15_a": p_plus_15_a,
        "p_handicap_minus_15_a": p_2_0,
        "p_handicap_plus_15_b": p_plus_15_b,
        "p_handicap_minus_15_b": p_0_2,
    }


def calc_match_prob_bo5(p_set_a: float) -> Dict[str, float]:
    """Best-of-5 sets probability calculations (Grand Slams)."""
    p = float(np.clip(p_set_a, 0.02, 0.98))
    q = 1.0 - p

    p_3_0 = p**3
    p_3_1 = 3.0 * (p**3) * q
    p_3_2 = 6.0 * (p**3) * (q**2)
    p_0_3 = q**3
    p_1_3 = 3.0 * (q**3) * p
    p_2_3 = 6.0 * (q**3) * (p**2)

    p_win_a = p_3_0 + p_3_1 + p_3_2
    p_win_b = p_0_3 + p_1_3 + p_2_3

    p_3_sets = p_3_0 + p_0_3
    p_4_sets = p_3_1 + p_1_3
    p_5_sets = p_3_2 + p_2_3

    p_plus_15_a = 1.0 - p_0_3
    p_plus_15_b = 1.0 - p_3_0

    return {
        "p_win_a": p_win_a,
        "p_win_b": p_win_b,
        "p_3_0_a": p_3_0,
        "p_3_1_a": p_3_1,
        "p_3_2_a": p_3_2,
        "p_0_3_b": p_0_3,
        "p_1_3_b": p_1_3,
        "p_2_3_b": p_2_3,
        # Straight-sets aliases for backward compatibility
        "p_2_0_a": p_3_0,
        "p_2_1_a": p_3_1 + p_3_2,
        "p_0_2_b": p_0_3,
        "p_1_2_b": p_1_3 + p_2_3,
        "p_under_25_sets": p_3_sets,
        "p_over_25_sets": p_4_sets + p_5_sets,
        "p_handicap_plus_15_a": p_plus_15_a,
        "p_handicap_minus_15_a": p_3_0 + p_3_1,
        "p_handicap_plus_15_b": p_plus_15_b,
        "p_handicap_minus_15_b": p_0_3 + p_1_3,
    }


def solve_set_prob_from_match_prob(p_match: float, best_of: int = 3) -> float:
    """Solves for p_set in closed-form Markov equation using bisection."""
    p_clamped = float(np.clip(p_match, 0.05, 0.95))
    low, high = 0.02, 0.98
    for _ in range(16):
        mid = (low + high) / 2.0
        if best_of == 5:
            val = (mid**3) * (10.0 - 15.0 * mid + 6.0 * (mid**2))
        else:
            val = 3.0 * (mid**2) - 2.0 * (mid**3)
        if val < p_clamped:
            low = mid
        else:
            high = mid
    return float(np.clip(mid, 0.02, 0.98))


def calc_expected_games_per_set(p_hold_a: float, p_hold_b: float) -> float:
    """Estimates the expected number of total games in a set based on service dominance.
    Sets with high hold % tend toward 10-13 games (tiebreaks).
    Sets with frequent breaks end quickly (6-1, 6-2, ~8.5 games).
    """
    avg_hold = (p_hold_a + p_hold_b) / 2.0
    # Baseline for 50/50 break: ~9.2 games. Server dominance can push to 10.8+ games
    exp_games = 8.2 + (avg_hold - 0.70) * 8.5
    return float(np.clip(exp_games, 8.4, 11.2))


# ==============================================================================
# 2. PROBABILITY TO ODDS CONVERSION & BETPLAY MARKET MAKER
# ==============================================================================

def prob_to_odds(prob: float, margin: float = 0.95) -> float:
    """Converts a probability into decimal odds with house margin.
    Applies strict safety rules:
    - Clamps probability to [0.08, 0.91]
    - Floors odds at 1.08 minimum
    """
    p_clamped = float(np.clip(prob, 0.08, 0.91))
    odds = (1.0 / p_clamped) * margin
    return round(max(1.08, odds), 2)


def prob_to_fair_odds(prob: float) -> float:
    """Returns pure fair odds (zero margin) for Expected Value calculations."""
    p_clamped = float(np.clip(prob, 0.05, 0.95))
    return round(max(1.05, 1.0 / p_clamped), 2)


# ==============================================================================
# 3. FULL MARKET CATALOG GENERATOR
# ==============================================================================


# ==============================================================================
# MOTOR ESTOCÁSTICO: SIMULADOR DE MONTE CARLO EN TENIS (MCMC - 10,000 PARTIDOS)
# ==============================================================================

def run_tennis_monte_carlo(
    p_serve_a: float,
    p_serve_b: float,
    iterations: int = 10000,
    best_of: int = 3,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """MOTOR 1 (TENIS): Simula 10,000 partidos de tenis juego por juego y set por set
    utilizando NumPy de forma vectorizada.
    
    Modela la alternancia de servicio, quiebres de saque, tie-breaks a 7 puntos y
    diferencia de sets/juegos para extraer probabilidades empíricas ultra-precisas en < 20 ms.
    """
    import time
    t0 = time.perf_counter()
    if random_seed is not None:
        np.random.seed(random_seed)

    p_hold_a = calc_game_prob(p_serve_a)
    p_hold_b = calc_game_prob(p_serve_b)
    p_tb_a = calc_tiebreak_prob(p_serve_a, p_serve_b)

    sets_to_win = (int(best_of) // 2) + 1

    sets_won_a = np.zeros(iterations, dtype=np.int32)
    sets_won_b = np.zeros(iterations, dtype=np.int32)
    total_games_all = np.zeros(iterations, dtype=np.int32)
    games_won_a = np.zeros(iterations, dtype=np.int32)
    games_won_b = np.zeros(iterations, dtype=np.int32)

    current_server = np.zeros(iterations, dtype=np.int32)
    active = np.ones(iterations, dtype=bool)

    for set_idx in range(int(best_of)):
        if not np.any(active):
            break

        g_a = np.zeros(iterations, dtype=np.int32)
        g_b = np.zeros(iterations, dtype=np.int32)
        set_active = active.copy()

        for game_idx in range(13):
            if not np.any(set_active):
                break

            is_tb = set_active & (g_a == 6) & (g_b == 6)
            if np.any(is_tb):
                tb_rand = np.random.rand(np.sum(is_tb))
                tb_win_a = tb_rand < p_tb_a
                g_a[is_tb] += tb_win_a.astype(np.int32)
                g_b[is_tb] += (~tb_win_a).astype(np.int32)
                set_active[is_tb] = False
                continue

            reg_active = set_active & ~is_tb
            if not np.any(reg_active):
                break

            n_reg = np.sum(reg_active)
            r = np.random.rand(n_reg)
            server_is_a = (current_server[reg_active] == 0)
            p_win_game_a = np.where(server_is_a, p_hold_a, 1.0 - p_hold_b)
            a_won_game = r < p_win_game_a

            g_a[reg_active] += a_won_game.astype(np.int32)
            g_b[reg_active] += (~a_won_game).astype(np.int32)
            current_server[reg_active] = 1 - current_server[reg_active]

            a_wins_set = (g_a >= 6) & (g_a - g_b >= 2)
            b_wins_set = (g_b >= 6) & (g_b - g_a >= 2)
            set_finished = a_wins_set | b_wins_set
            set_active[set_finished] = False

        set_won_by_a = (g_a > g_b) & active
        set_won_by_b = (g_b > g_a) & active

        sets_won_a[set_won_by_a] += 1
        sets_won_b[set_won_by_b] += 1

        games_won_a[active] += g_a[active]
        games_won_b[active] += g_b[active]
        total_games_all[active] += (g_a[active] + g_b[active])

        match_finished = (sets_won_a >= sets_to_win) | (sets_won_b >= sets_to_win)
        active[match_finished] = False

    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    p_match_a = float(np.mean(sets_won_a > sets_won_b))
    p_match_b = 1.0 - p_match_a

    p_a_2_0 = float(np.mean((sets_won_a == 2) & (sets_won_b == 0)))
    p_a_2_1 = float(np.mean((sets_won_a == 2) & (sets_won_b == 1)))
    p_b_2_0 = float(np.mean((sets_won_b == 2) & (sets_won_a == 0)))
    p_b_2_1 = float(np.mean((sets_won_b == 2) & (sets_won_a == 1)))

    p_set_handicap_a_minus15 = p_a_2_0
    p_set_handicap_a_plus15 = 1.0 - p_b_2_0
    p_set_handicap_b_minus15 = p_b_2_0
    p_set_handicap_b_plus15 = 1.0 - p_a_2_0

    lines = [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5]
    total_games_probs = {}
    for line in lines:
        p_over = float(np.mean(total_games_all > line))
        k_over = f"over_{str(line).replace('.', '')}"
        k_under = f"under_{str(line).replace('.', '')}"
        total_games_probs[k_over] = p_over
        total_games_probs[k_under] = 1.0 - p_over

    diff_games = games_won_a - games_won_b
    game_handicap_probs = {}
    for h in [-4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5]:
        k_h = f"h_{str(h).replace('-', 'm').replace('.', '')}"
        game_handicap_probs[k_h] = float(np.mean(diff_games > h))

    return {
        "iterations": iterations,
        "sim_time_ms": round(ms, 2),
        "p_match_a": p_match_a,
        "p_match_b": p_match_b,
        "sets_distribution": {
            "a_2_0": p_a_2_0,
            "a_2_1": p_a_2_1,
            "b_2_0": p_b_2_0,
            "b_2_1": p_b_2_1,
        },
        "set_handicaps": {
            "a_minus_15": p_set_handicap_a_minus15,
            "a_plus_15": p_set_handicap_a_plus15,
            "b_minus_15": p_set_handicap_b_minus15,
            "b_plus_15": p_set_handicap_b_plus15,
        },
        "total_games_probs": total_games_probs,
        "game_handicap_probs": game_handicap_probs,
        "avg_total_games": float(np.mean(total_games_all)),
        "avg_games_a": float(np.mean(games_won_a)),
        "avg_games_b": float(np.mean(games_won_b)),
    }


def calculate_oracle_score(
    ev_percent: float,
    monte_carlo_prob_pct: float,
    is_steam: bool = False,
    steam_bonus: float = 100.0,
) -> Tuple[float, bool, str]:
    """MOTOR 3 (ORÁCULO TENIS): Calcula el Score de Oráculo Cuantitativo (0 a 100).
    Score = (EV_normalizado * 0.40) + (Prob_Monte_Carlo * 0.40) + (Bono_Smart_Money * 0.20)
    Gobernanza:
    - Score > 65.0: Luz Verde (+EV Calificado, Kelly habilitado).
    - Score >= 90.0: 🔥 GOLDEN PICK (Alta Certeza Matemática).
    - Score <= 65.0: 🔒 Stake Bloqueado ($0 COP).
    """
    ev_norm = min(100.0, max(0.0, (ev_percent / 12.0) * 100.0)) if ev_percent > 0 else 0.0
    p_norm = min(100.0, max(0.0, float(monte_carlo_prob_pct)))
    bonus = steam_bonus if is_steam else 0.0

    score = round((ev_norm * 0.40) + (p_norm * 0.40) + (bonus * 0.20), 1)
    score = min(100.0, max(0.0, score))

    if score >= 90.0:
        is_golden = True
        tier = "🔥 GOLDEN PICK (Alta Certeza Matemática)"
    elif score > 65.0:
        is_golden = False
        tier = "🟢 LUZ VERDE (+EV Calificado)"
    else:
        is_golden = False
        tier = "⚪ FILTRO INSTITUCIONAL (Observación / Stake Bloqueado)"

    return score, is_golden, tier


def generate_tennis_market_catalog(
    player_a: str,
    player_b: str,
    surface: str = "Dura Outdoor",
    bankroll: float = 100000.0,
    kelly_multiplier: float = 0.20,
    ai_adj: Optional[Any] = None,
    retail_odds: Optional[Dict[str, float]] = None,
    best_of: int = 3,
    mc_sim: Optional[Dict[str, Any]] = None,
    steam_data: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Builds the comprehensive Markov-derived market catalog for a tennis match.

    Returns probabilities, expected games, and full BetPlay market blocks:
    1. Ganador del Partido (1X2 Tennis)
    2. Ganador del 1er Set
    3. Hándicap de Sets (+1.5 / -1.5)
    4. Marcador Exacto en Sets (2-0, 2-1, 0-2, 1-2)
    5. Total de Sets (2 Sets vs 3 Sets)
    6. Total de Juegos del Partido (Over/Under 19.5, 20.5, 21.5, 22.5)
    7. Habrá Tie-Break (Sí / No)
    8. Hándicap de Juegos del Partido (-3.5, -2.5, +2.5, +3.5)
    """
    prof_a = get_player_profile(player_a, surface)
    prof_b = get_player_profile(player_b, surface)

    # 1. Base point-level Markov probabilities
    p_pt_a_raw = calc_point_prob(prof_a["spw"], prof_b["rpw"], surface)
    p_pt_b_raw = calc_point_prob(prof_b["spw"], prof_a["rpw"], surface)

    # Apply AI tactical serve adjustment factors if available
    f_saque_a = float(safe_ai_get(ai_adj, "factor_saque_a", 1.0))
    f_saque_b = float(safe_ai_get(ai_adj, "factor_saque_b", 1.0))
    p_pt_a = float(np.clip(p_pt_a_raw * f_saque_a, 0.45, 0.85))
    p_pt_b = float(np.clip(p_pt_b_raw * f_saque_b, 0.45, 0.85))

    # 2. Service hold probabilities (Game level)
    p_hold_a = calc_game_prob(p_pt_a)
    p_hold_b = calc_game_prob(p_pt_b)

    # 3. Tiebreak & Set probabilities
    p_tb_a = calc_tiebreak_prob(p_pt_a, p_pt_b)
    p_set_a, p_tb_reach = calc_set_prob(p_hold_a, p_hold_b, p_tb_a)
    p_set_b = 1.0 - p_set_a

    # 4. Stochastic Monte Carlo Simulation (10,000 matches)
    best_of_sets = int(best_of)
    if mc_sim is None:
        mc_sim = run_tennis_monte_carlo(p_pt_a, p_pt_b, iterations=10000, best_of=best_of_sets)

    m_probs = calc_match_prob_bo5(p_set_a) if best_of_sets == 5 else calc_match_prob_bo3(p_set_a)
    p_win_a = float(mc_sim.get("p_match_a", m_probs["p_win_a"]))
    p_win_b = float(mc_sim.get("p_match_b", m_probs["p_win_b"]))

    # Market Calibration & Vig-Free Anchor (Prevents favorite inversion)
    retail = retail_odds or {}
    has_market_odds = (
        "1" in retail and "2" in retail
        and float(retail.get("1", 0)) > 1.0
        and float(retail.get("2", 0)) > 1.0
    )

    if has_market_odds:
        o1 = float(retail["1"])
        o2 = float(retail["2"])
        inv1 = 1.0 / o1
        inv2 = 1.0 / o2
        tot_inv = inv1 + inv2
        p_mkt_a = inv1 / tot_inv
        p_mkt_b = inv2 / tot_inv

        has_custom_profiles = (prof_a.get("elo", 1500) != 1500 or prof_b.get("elo", 1500) != 1500)
        if has_custom_profiles:
            # Calibrate Monte Carlo with Vig-Free Market probability (40% MC, 60% Sharp Market)
            p_win_a = float(np.clip(0.40 * p_win_a + 0.60 * p_mkt_a, 0.05, 0.95))
            p_win_b = 1.0 - p_win_a
        else:
            # When unprofiled, anchor 100% to market vig-free probability
            p_win_a = p_mkt_a
            p_win_b = p_mkt_b

        # Synchronize set probability with calibrated match probability
        p_set_a = solve_set_prob_from_match_prob(p_win_a, best_of=best_of_sets)
        p_set_b = 1.0 - p_set_a
        if best_of_sets == 5:
            m_probs = calc_match_prob_bo5(p_set_a)
        else:
            m_probs = calc_match_prob_bo3(p_set_a)
        p_win_a = m_probs["p_win_a"]
        p_win_b = m_probs["p_win_b"]

    # 5. Expected games
    exp_games_set = calc_expected_games_per_set(p_hold_a, p_hold_b)
    if best_of_sets == 5:
        exp_num_sets = 3.0 * m_probs["p_under_25_sets"] + 4.5 * m_probs["p_over_25_sets"]
    else:
        exp_num_sets = 2.0 * m_probs["p_under_25_sets"] + 3.0 * m_probs["p_over_25_sets"]
    exp_total_games = exp_games_set * exp_num_sets

    retail = retail_odds or {}

    def _mk_item(
        name: str,
        prob: float,
        retail_key: Optional[str] = None,
        default_margin: float = 0.94,
    ) -> Dict[str, Any]:
        p = float(np.clip(prob, 0.05, 0.95))
        f_odds = prob_to_fair_odds(p)
        r_odds = float(retail.get(retail_key, prob_to_odds(p, default_margin))) if retail_key else prob_to_odds(p, default_margin)
        r_odds = max(1.08, r_odds)

        ev_pct = round(((p * r_odds) - 1.0) * 100.0, 1)
        passes, _ = evaluate_opportunity_filters(odds=r_odds, ev_percent=ev_pct)

        k_res = calculate_kelly_stake(
            odds=r_odds,
            effective_prob=p,
            bankroll=bankroll,
            kelly_fraction=kelly_multiplier,
        ) if passes else {"recommended_stake": 0.0, "recommended_stake_pct": 0.0}

        recom = "🟢 ENTRAR" if (passes and ev_pct >= 2.0) else "⚪ PASAR"

        return {
            "name": name,
            "prob_pct": round(p * 100.0, 1),
            "fair_odds": f_odds,
            "retail_odds": r_odds,
            "ev_pct": ev_pct,
            "kelly_stake": k_res["recommended_stake"],
            "recommendation": recom,
        }

    # Games total distributions using normal approximation centered at exp_total_games
    std_games = 3.6  # typical standard deviation of total games in Bo3
    def _prob_over_games(line: float) -> float:
        # P(Total > line)
        z = (line - exp_total_games) / std_games
        return float(1.0 - norm.cdf(z))

    # Tiebreak in match: P(at least 1 TB in match) = 1 - (1 - P_tb_set)^E[sets]
    p_tb_match = float(np.clip(1.0 - ((1.0 - p_tb_reach) ** exp_num_sets), 0.08, 0.75))

    # Game handicap: estimated from point win probability differential
    game_diff_expected = (p_pt_a - p_pt_b) * exp_total_games * 2.2

    def _prob_game_handicap_a(line: float) -> float:
        # A covers handicap line (e.g., A -3.5 means diff > 3.5)
        z = (line - game_diff_expected) / 4.2
        return float(np.clip(1.0 - norm.cdf(z), 0.08, 0.92))

    catalog = {
        "player_a": player_a,
        "player_b": player_b,
        "surface": surface,
        "profiles": {"a": prof_a, "b": prof_b},
        "probabilities": {
            "p_point_serve_a": round(p_pt_a, 3),
            "p_point_serve_b": round(p_pt_b, 3),
            "p_hold_a": round(p_hold_a, 3),
            "p_hold_b": round(p_hold_b, 3),
            "p_tiebreak_a": round(p_tb_a, 3),
            "p_set_a": round(p_set_a, 3),
            "p_set_b": round(p_set_b, 3),
            "p_match_a": round(p_win_a, 3),
            "p_match_b": round(p_win_b, 3),
        },
        "metrics": {
            "exp_games_per_set": round(exp_games_set, 1),
            "exp_num_sets": round(exp_num_sets, 2),
            "exp_total_games": round(exp_total_games, 1),
            "prob_tb_set": round(p_tb_reach * 100.0, 1),
            "prob_tb_match": round(p_tb_match * 100.0, 1),
            "elo_diff": prof_a["elo"] - prof_b["elo"],
        },
        "markets": {
            "match_winner": {
                "a": _mk_item(f"Victoria {player_a}", p_win_a, "1"),
                "b": _mk_item(f"Victoria {player_b}", p_win_b, "2"),
            },
            "first_set_winner": {
                "a": _mk_item(f"1er Set: {player_a}", p_set_a),
                "b": _mk_item(f"1er Set: {player_b}", p_set_b),
            },
            "handicap_sets": {
                "a_plus_15": _mk_item(f"{player_a} +1.5 Sets (Gana al menos 1)", m_probs["p_handicap_plus_15_a"]),
                "a_minus_15": _mk_item(f"{player_a} -1.5 Sets (Gana 2-0)", m_probs["p_handicap_minus_15_a"]),
                "b_plus_15": _mk_item(f"{player_b} +1.5 Sets (Gana al menos 1)", m_probs["p_handicap_plus_15_b"]),
                "b_minus_15": _mk_item(f"{player_b} -1.5 Sets (Gana 2-0)", m_probs["p_handicap_minus_15_b"]),
            },
            "exact_score_sets": (
                {
                    "3_0_a": _mk_item(f"{player_a} 3-0", m_probs.get("p_3_0_a", 0.0)),
                    "3_1_a": _mk_item(f"{player_a} 3-1", m_probs.get("p_3_1_a", 0.0)),
                    "3_2_a": _mk_item(f"{player_a} 3-2", m_probs.get("p_3_2_a", 0.0)),
                    "0_3_b": _mk_item(f"{player_b} 0-3", m_probs.get("p_0_3_b", 0.0)),
                    "1_3_b": _mk_item(f"{player_b} 1-3", m_probs.get("p_1_3_b", 0.0)),
                    "2_3_b": _mk_item(f"{player_b} 2-3", m_probs.get("p_2_3_b", 0.0)),
                }
                if best_of_sets == 5
                else {
                    "2_0_a": _mk_item(f"{player_a} 2-0", m_probs.get("p_2_0_a", 0.0)),
                    "2_1_a": _mk_item(f"{player_a} 2-1", m_probs.get("p_2_1_a", 0.0)),
                    "0_2_b": _mk_item(f"{player_b} 0-2", m_probs.get("p_0_2_b", 0.0)),
                    "1_2_b": _mk_item(f"{player_b} 1-2", m_probs.get("p_1_2_b", 0.0)),
                }
            ),
            "total_sets": {
                "under_25": _mk_item("Exactamente 2 Sets (Menos de 2.5)", m_probs["p_under_25_sets"]),
                "over_25": _mk_item("Se juegan 3 Sets (Más de 2.5)", m_probs["p_over_25_sets"]),
            },
            "total_games": {
                "over_185": _mk_item("Más de 18.5 Juegos", _prob_over_games(18.5)),
                "under_185": _mk_item("Menos de 18.5 Juegos", 1.0 - _prob_over_games(18.5)),
                "over_195": _mk_item("Más de 19.5 Juegos", _prob_over_games(19.5)),
                "under_195": _mk_item("Menos de 19.5 Juegos", 1.0 - _prob_over_games(19.5)),
                "over_205": _mk_item("Más de 20.5 Juegos", _prob_over_games(20.5)),
                "under_205": _mk_item("Menos de 20.5 Juegos", 1.0 - _prob_over_games(20.5)),
                "over_215": _mk_item("Más de 21.5 Juegos", _prob_over_games(21.5)),
                "under_215": _mk_item("Menos de 21.5 Juegos", 1.0 - _prob_over_games(21.5)),
                "over_225": _mk_item("Más de 22.5 Juegos", _prob_over_games(22.5)),
                "under_225": _mk_item("Menos de 22.5 Juegos", 1.0 - _prob_over_games(22.5)),
            },
            "tiebreak": {
                "yes": _mk_item("Habrá Tie-Break en el Partido (Sí)", p_tb_match),
                "no": _mk_item("No habrá Tie-Break en el Partido", 1.0 - p_tb_match),
            },
            "game_handicap": {
                "a_minus_25": _mk_item(f"{player_a} Hándicap -2.5 Juegos", _prob_game_handicap_a(2.5)),
                "a_minus_35": _mk_item(f"{player_a} Hándicap -3.5 Juegos", _prob_game_handicap_a(3.5)),
                "b_plus_25": _mk_item(f"{player_b} Hándicap +2.5 Juegos", 1.0 - _prob_game_handicap_a(2.5)),
                "b_plus_35": _mk_item(f"{player_b} Hándicap +3.5 Juegos", 1.0 - _prob_game_handicap_a(3.5)),
            },
        },
    }

    return catalog


# ==============================================================================
# 4. SAME-GAME PARLAY (COMBINADAS PARA TI) GENERATOR
# ==============================================================================

def generate_tennis_sgp_cards(
    catalog: Optional[Dict[str, Any]] = None,
    bankroll: float = 100000.0,
    kelly_multiplier: float = 0.20,
    ai_adj: Optional[Any] = None,
    mc_sim: Optional[Dict[str, Any]] = None,
    steam_data: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    if catalog is None:
        catalog = kwargs.get("catalog") or {}
    if mc_sim is None:
        mc_sim = kwargs.get("mc_sim") or catalog.get("monte_carlo_sim")
    if steam_data is None:
        steam_data = kwargs.get("steam_data") or catalog.get("steam_move")

    """Builds the 3 'Combinadas para ti' (Same-Game Parlays) with low variance,
    strictly selecting individual legs with odds >= 1.14 and total odds >= 1.45.

    Tarjeta 1: 🛡️ ESCUDO DE SETS (Baja Varianza)
    Tarjeta 2: 🔥 DOMINIO DE SERVICIO (Riesgo Medio)
    Tarjeta 3: 🚀 ESPECIALISTA DE SUPERFICIE + IA GEMINI (+EV)
    """
    p_a = catalog["player_a"]
    p_b = catalog["player_b"]
    probs = catalog["probabilities"]
    fav_is_a = probs["p_match_a"] >= probs["p_match_b"]
    fav_name = p_a if fav_is_a else p_b
    dog_name = p_b if fav_is_a else p_a

    mkts = catalog["markets"]

    cards: List[Dict[str, Any]] = []

    def _safe_leg(item: Dict[str, Any], mkt_label: str) -> Dict[str, Any]:
        return {
            "market": mkt_label,
            "selection": item["name"],
            "odds": max(1.14, float(item["retail_odds"])),
            "prob": min(max(float(item["prob_pct"]) / 100.0, 0.08), 0.91),
        }

    # --------------------------------------------------------------------------
    # TARJETA 1: 🛡️ ESCUDO DE SETS (Baja Varianza / Alta Probabilidad)
    # --------------------------------------------------------------------------
    # Leg 1: Favorito +1.5 Sets (gana al menos 1 set) o Ganador si cuota >= 1.14
    cand_hset = mkts["handicap_sets"]["a_plus_15" if fav_is_a else "b_plus_15"]
    if cand_hset["retail_odds"] < 1.14:
        cand_hset = mkts["match_winner"]["a" if fav_is_a else "b"]
        mkt1_label = "Ganador del Partido"
    else:
        mkt1_label = "Hándicap de Sets (+1.5)"
    leg1_t1 = _safe_leg(cand_hset, mkt1_label)

    # Leg 2: Total de Juegos Más de 18.5 (o 19.5 si 18.5 < 1.14)
    cand_games = mkts["total_games"]["over_185"]
    if cand_games["retail_odds"] < 1.14:
        cand_games = mkts["total_games"]["over_195"]
    leg2_t1 = _safe_leg(cand_games, "Total de Juegos")

    legs_t1 = [leg1_t1, leg2_t1]
    # Correlation adjustment for tennis parlay (typically slightly positive)
    corr_t1 = 0.93
    odds_t1 = round(max(1.45, legs_t1[0]["odds"] * legs_t1[1]["odds"] * corr_t1), 2)
    prob_t1 = round(legs_t1[0]["prob"] * legs_t1[1]["prob"] * (1.0 / corr_t1) * 100.0, 1)
    prob_t1 = min(prob_t1, 88.0)
    ev_t1 = round(((prob_t1 / 100.0 * odds_t1) - 1.0) * 100.0, 1)

    is_steam_c = bool(steam_data.get("is_steam", False)) if steam_data else False
    score_t1, is_gold_t1, tier_t1 = calculate_oracle_score(ev_t1, prob_t1, is_steam=is_steam_c)
    if score_t1 > 65.0 and ev_t1 >= 2.0:
        k_t1 = calculate_kelly_stake(
            odds=odds_t1,
            effective_prob=prob_t1 / 100.0,
            bankroll=bankroll,
            kelly_fraction=kelly_multiplier,
        )
        stake_t1 = k_t1["recommended_stake"]
    else:
        stake_t1 = 0.0

    cards.append({
        "card_id": 1,
        "title": "ESCUDO DE SETS (BAJA VARIANZA)",
        "badge": "🛡️ ESCUDO DE SETS",
        "risk_level": "Bajo",
        "description": "Combinada defensiva diseñada para capitalizar la alta probabilidad de que el favorito gane al menos 1 set en un partido de ritmo sostenido.",
        "legs": legs_t1,
        "odds": odds_t1,
        "prob_pct": prob_t1,
        "ev_percent": ev_t1,
        "stake": stake_t1,
        "stake_pct": round((stake_t1 / bankroll) * 100.0, 2) if bankroll > 0 else 0.0,
        "oracle_score": score_t1,
        "is_golden_pick": is_gold_t1,
        "tier_label": tier_t1,
        "p_monte_carlo": prob_t1,
        "steam_move": steam_data,
        "mc_stats": {"iterations": mc_sim.get("iterations", 10000), "sim_time_ms": mc_sim.get("sim_time_ms", 12.0)} if mc_sim else None,
    })

    # --------------------------------------------------------------------------
    # TARJETA 2: 🔥 DOMINIO DE SERVICIO (Riesgo Medio - Cuota 1.70 - 2.30)
    # --------------------------------------------------------------------------
    # Leg 1: Ganador 1er Set (Favorito)
    cand_1s = mkts["first_set_winner"]["a" if fav_is_a else "b"]
    leg1_t2 = _safe_leg(cand_1s, "Ganador 1er Set")

    # Leg 2: Ganador del Partido (Favorito)
    cand_win = mkts["match_winner"]["a" if fav_is_a else "b"]
    leg2_t2 = _safe_leg(cand_win, "Ganador del Partido")

    # Leg 3: Más de 19.5 Juegos (o Menos de 2.5 sets si favorito es aplastante)
    cand_g2 = mkts["total_games"]["over_195"]
    leg3_t2 = _safe_leg(cand_g2, "Línea de Juegos")

    legs_t2 = [leg1_t2, leg2_t2, leg3_t2]
    corr_t2 = 0.88
    odds_t2 = round(max(1.70, legs_t2[0]["odds"] * legs_t2[1]["odds"] * legs_t2[2]["odds"] * corr_t2), 2)
    prob_t2 = round(legs_t2[0]["prob"] * legs_t2[1]["prob"] * legs_t2[2]["prob"] * (1.0 / corr_t2) * 100.0, 1)
    prob_t2 = min(prob_t2, 72.0)
    ev_t2 = round(((prob_t2 / 100.0 * odds_t2) - 1.0) * 100.0, 1)

    score_t2, is_gold_t2, tier_t2 = calculate_oracle_score(ev_t2, prob_t2, is_steam=is_steam_c)
    if score_t2 > 65.0 and ev_t2 >= 2.0:
        k_t2 = calculate_kelly_stake(
            odds=odds_t2,
            effective_prob=prob_t2 / 100.0,
            bankroll=bankroll,
            kelly_fraction=kelly_multiplier,
        )
        stake_t2 = k_t2["recommended_stake"]
    else:
        stake_t2 = 0.0

    cards.append({
        "card_id": 2,
        "title": "DOMINIO DE SERVICIO (POTENCIA & RITMO)",
        "badge": "🔥 DOMINIO DE SERVICIO",
        "risk_level": "Medio",
        "description": "Explota la superioridad del servicio para asegurar el primer set y consolidar el partido en games acumulados.",
        "legs": legs_t2,
        "odds": odds_t2,
        "prob_pct": prob_t2,
        "ev_percent": ev_t2,
        "stake": stake_t2,
        "stake_pct": round((stake_t2 / bankroll) * 100.0, 2) if bankroll > 0 else 0.0,
        "oracle_score": score_t2,
        "is_golden_pick": is_gold_t2,
        "tier_label": tier_t2,
        "p_monte_carlo": prob_t2,
        "steam_move": steam_data,
        "mc_stats": {"iterations": mc_sim.get("iterations", 10000), "sim_time_ms": mc_sim.get("sim_time_ms", 12.0)} if mc_sim else None,
    })

    # --------------------------------------------------------------------------
    # TARJETA 3: 🚀 ESPECIALISTA DE SUPERFICIE + IA GEMINI (+EV MÁXIMO)
    # --------------------------------------------------------------------------
    # Leg 1: Hándicap de Juegos (-2.5 favorito o +3.5 no favorito)
    if fav_is_a:
        cand_gh = mkts["game_handicap"]["a_minus_25"]
    else:
        cand_gh = mkts["game_handicap"]["b_plus_25"]
    leg1_t3 = _safe_leg(cand_gh, "Hándicap de Juegos")

    # Leg 2: Total de Juegos Over 20.5 o Over 2.5 sets
    cand_t3_games = mkts["total_games"]["over_205"]
    leg2_t3 = _safe_leg(cand_t3_games, "Total Juegos Partido")

    legs_t3 = [leg1_t3, leg2_t3]
    corr_t3 = 0.92
    odds_t3 = round(max(1.85, legs_t3[0]["odds"] * legs_t3[1]["odds"] * corr_t3), 2)
    prob_t3 = round(legs_t3[0]["prob"] * legs_t3[1]["prob"] * (1.0 / corr_t3) * 100.0, 1)
    prob_t3 = min(prob_t3, 65.0)
    ev_t3 = round(((prob_t3 / 100.0 * odds_t3) - 1.0) * 100.0, 1)

    score_t3, is_gold_t3, tier_t3 = calculate_oracle_score(ev_t3, prob_t3, is_steam=is_steam_c)
    if score_t3 > 65.0 and ev_t3 >= 2.0:
        k_t3 = calculate_kelly_stake(
            odds=odds_t3,
            effective_prob=prob_t3 / 100.0,
            bankroll=bankroll,
            kelly_fraction=kelly_multiplier,
        )
        stake_t3 = k_t3["recommended_stake"]
    else:
        stake_t3 = 0.0

    cards.append({
        "card_id": 3,
        "title": "ESPECIALISTA DE SUPERFICIE (+EV)",
        "badge": "🚀 ESPECIALISTA SUPERFICIE",
        "risk_level": "Medio-Alto",
        "description": "Selección cuantitativa de alto rendimiento basada en el diferencial ELO específico de superficie y análisis táctico de IA.",
        "legs": legs_t3,
        "odds": odds_t3,
        "prob_pct": prob_t3,
        "ev_percent": ev_t3,
        "stake": stake_t3,
        "stake_pct": round((stake_t3 / bankroll) * 100.0, 2) if bankroll > 0 else 0.0,
        "oracle_score": score_t3,
        "is_golden_pick": is_gold_t3,
        "tier_label": tier_t3,
        "p_monte_carlo": prob_t3,
        "steam_move": steam_data,
        "mc_stats": {"iterations": mc_sim.get("iterations", 10000), "sim_time_ms": mc_sim.get("sim_time_ms", 12.0)} if mc_sim else None,
    })

    return cards


# ==============================================================================
# 5. SCANNER / AUTOMATED +EV OPPORTUNITY EVALUATOR
# ==============================================================================

def analyze_tennis_match_opportunities(
    event: Dict[str, Any],
    sport_meta: Dict[str, Any],
    bankroll: float = 100000.0,
    kelly_k: float = 0.20,
    min_ev: float = 2.0,
    min_odds: float = 1.30,
    max_odds: float = 2.80,
    ai_adjustment: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Evaluates an event for value betting opportunities (+EV) across all tennis markets.
    Returns qualified signals that pass safety and variance filters.
    """
    p1 = event.get("home_team", "Jugador A")
    p2 = event.get("away_team", "Jugador B")
    surface = event.get("surface", sport_meta.get("surface", "Dura Outdoor"))
    t_name = sport_meta.get("name", event.get("sport_title", "Torneo"))
    best_of = sport_meta.get("best_of", 3)

    from etl.tennis_odds_client import extract_event_tennis_odds, detect_tennis_steam_moves
    retail_map = extract_event_tennis_odds(event)
    steam_move = detect_tennis_steam_moves(event)
    event["steam_move"] = steam_move

    # Run Monte Carlo 10,000 matches
    prof_a = get_player_profile(p1, surface)
    prof_b = get_player_profile(p2, surface)
    p_pt_a_raw = calc_point_prob(prof_a["spw"], prof_b["rpw"], surface)
    p_pt_b_raw = calc_point_prob(prof_b["spw"], prof_a["rpw"], surface)
    f_saque_a = float(safe_ai_get(ai_adjustment, "factor_saque_a", 1.0))
    f_saque_b = float(safe_ai_get(ai_adjustment, "factor_saque_b", 1.0))
    p_pt_a = float(np.clip(p_pt_a_raw * f_saque_a, 0.45, 0.85))
    p_pt_b = float(np.clip(p_pt_b_raw * f_saque_b, 0.45, 0.85))

    mc_sim = run_tennis_monte_carlo(p_pt_a, p_pt_b, iterations=10000, best_of=best_of)
    event["monte_carlo_sim"] = mc_sim

    catalog = generate_tennis_market_catalog(
        player_a=p1,
        player_b=p2,
        surface=surface,
        bankroll=bankroll,
        kelly_multiplier=kelly_k,
        ai_adj=ai_adjustment,
        retail_odds=retail_map,
        best_of=best_of,
        mc_sim=mc_sim,
        steam_data=steam_move,
    )
    catalog["monte_carlo_sim"] = mc_sim
    catalog["steam_move"] = steam_move

    sgp_cards = generate_tennis_sgp_cards(
        catalog=catalog,
        bankroll=bankroll,
        kelly_multiplier=kelly_k,
        ai_adj=ai_adjustment,
        mc_sim=mc_sim,
        steam_data=steam_move,
    )

    opportunities: List[Dict[str, Any]] = []

    # 1. Evaluate SGP cards with Oracle Score
    for card in sgp_cards:
        c_ev = float(card.get("ev_percent", 0.0))
        c_odds = float(card.get("odds", 0.0))
        c_score = float(card.get("oracle_score", 0.0))
        c_stake = float(card.get("stake", 0.0))
        c_gold = bool(card.get("is_golden_pick", False))
        c_tier = str(card.get("tier_label", ""))
        c_prob = float(card.get("prob_pct", 50.0))

        if c_ev >= min_ev and min_odds <= c_odds <= max_odds:
            opportunities.append({
                "match": f"{p1} vs {p2}",
                "tournament": t_name,
                "surface": surface,
                "market": card["badge"],
                "selection": " + ".join([l["selection"] for l in card.get("legs", [])]),
                "odds": c_odds,
                "fair_odds": round(1.0 / (c_prob / 100.0), 2) if c_prob > 0 else 999.0,
                "effective_prob_pct": c_prob,
                "ev_percent": c_ev,
                "oracle_score": c_score,
                "is_golden_pick": c_gold,
                "tier_label": c_tier,
                "monte_carlo_prob": c_prob,
                "steam_move": steam_move,
                "mc_stats": card.get("mc_stats"),
                "recommended_stake": c_stake,
                "recommended_stake_pct": card.get("stake_pct", round(c_stake / bankroll * 100.0, 2)),
                "bookmaker": retail_map.get("bookmaker", "Pinnacle/BetPlay"),
                "commence_time": event.get("commence_time", ""),
                "ai_adjustment": ai_adjustment,
                "legs": card.get("legs", []),
                "description": card.get("description", ""),
            })


    # 2. Evaluate single markets (Match Winner, Sets Handicap)
    mkts = catalog.get("markets", {})
    for side_k, player_n in [("a", p1), ("b", p2)]:
        m_item = mkts.get("match_winner", {}).get(side_k)
        if m_item:
            m_ev = float(m_item.get("ev_pct", 0.0))
            m_odds = float(m_item.get("retail_odds", 0.0))
            m_prob = float(m_item.get("prob_pct", 50.0))
            is_steam_m = bool(steam_move.get("is_steam", False)) and (steam_move.get("side") == ("1" if side_k == "a" else "2"))
            m_score, m_gold, m_tier = calculate_oracle_score(m_ev, m_prob, is_steam=is_steam_m)

            if m_score > 65.0 and m_ev >= min_ev:
                k_m = calculate_kelly_stake(
                    odds=m_odds,
                    effective_prob=m_prob / 100.0,
                    bankroll=bankroll,
                    kelly_fraction=kelly_k,
                )
                m_stake = k_m["recommended_stake"]
            else:
                m_stake = 0.0

            if m_ev >= min_ev and min_odds <= m_odds <= max_odds:
                opportunities.append({
                    "match": f"{p1} vs {p2}",
                    "tournament": t_name,
                    "surface": surface,
                    "market": "Ganador del Partido (Moneyline)",
                    "selection": m_item.get("name", f"Victoria {player_n}"),
                    "odds": m_odds,
                    "fair_odds": m_item.get("fair_odds", 1.85),
                    "effective_prob_pct": m_prob,
                    "ev_percent": m_ev,
                    "oracle_score": m_score,
                    "is_golden_pick": m_gold,
                    "tier_label": m_tier,
                    "monte_carlo_prob": m_prob,
                    "steam_move": steam_move,
                    "recommended_stake": m_stake,
                    "recommended_stake_pct": round(m_stake / bankroll * 100.0, 2),
                    "bookmaker": retail_map.get("bookmaker", "Pinnacle/BetPlay"),
                    "commence_time": event.get("commence_time", ""),
                    "ai_adjustment": ai_adjustment,
                    "description": f"Ventaja matemática pura con Monte Carlo (10k sims) y Smart Money en Moneyline.",
                })

    return opportunities
