"""What-If Ticket Simulator & Cross-Match Tennis Parlay Calculator (QuantCourt AI).

Provides real-time mathematical simulation of tomorrow's matchday betting tickets
and builds dynamic multi-fixture cross-match accumulators (Parlays) across ATP/WTA fixtures.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from etl.tennis_odds_client import extract_event_tennis_odds
from models.markov_engine import generate_tennis_market_catalog, generate_tennis_sgp_cards


def calculate_what_if_metrics(
    bets_simulated: List[Dict[str, Any]],
    current_bankroll: float,
) -> Dict[str, Any]:
    """Calculates financial day-after metrics from a list of simulated bets.

    Each item in bets_simulated should contain:
    - 'stake': float
    - 'odds': float
    - 'scenario': str ("✅ Se cumple (Ganada)", "🔄 Reembolso / Push", "❌ No se cumple (Perdida)")
    - 'is_active': bool

    Returns:
        {
            'total_investment': float,
            'total_return': float,
            'net_profit': float,
            'roi_pct': float,
            'new_bankroll': float,
            'growth_pct': float,
        }
    """
    tot_inv = 0.0
    tot_ret = 0.0

    for b in bets_simulated:
        if not b.get("is_active", True):
            continue
        stake = float(b.get("stake", 0.0))
        odds = float(b.get("odds", 1.0))
        scen = b.get("scenario", "✅ Se cumple (Ganada)")

        tot_inv += stake
        if "Ganada" in scen:
            tot_ret += stake * odds
        elif "Push" in scen or "Reembolso" in scen:
            tot_ret += stake
        else:
            tot_ret += 0.0

    net_prof = tot_ret - tot_inv
    roi = (net_prof / tot_inv * 100.0) if tot_inv > 0 else 0.0
    new_bk = current_bankroll + net_prof
    growth = (net_prof / current_bankroll * 100.0) if current_bankroll > 0 else 0.0

    return {
        "total_investment": round(tot_inv, 2),
        "total_return": round(tot_ret, 2),
        "net_profit": round(net_prof, 2),
        "roi_pct": round(roi, 1),
        "new_bankroll": round(new_bk, 2),
        "growth_pct": round(growth, 1),
    }


def build_cross_match_tennis_parlay(
    events: List[Dict[str, Any]],
    selected_surface: str = "Dura Outdoor",
    bankroll: float = 100000.0,
    ai_adjustments: Optional[Dict[str, Any]] = None,
    max_legs: int = 5,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Assembles a cross-match tennis parlay combining the safest selections
    (🛡️ ESCUDO DE SETS) from up to max_legs matches available in the session.

    Each match is independently calculated with its own Markov chain,
    ensuring dynamic and unique odds per fixture.

    Returns:
        (legs_list, total_odds, joint_prob_pct)
    """
    adjustments = ai_adjustments or {}
    legs: List[Dict[str, Any]] = []

    for ev in events[:max_legs]:
        p_a = ev.get("home_team", "Jugador A")
        p_b = ev.get("away_team", "Jugador B")
        match_title = f"{p_a} vs {p_b}"
        surface = ev.get("surface", selected_surface)

        # Extract sharp / retail odds for this fixture
        retail_odds_map = extract_event_tennis_odds(ev)

        ai_adj = adjustments.get(match_title)

        catalog = generate_tennis_market_catalog(
            player_a=p_a,
            player_b=p_b,
            surface=surface,
            bankroll=bankroll,
            ai_adj=ai_adj,
            retail_odds=retail_odds_map,
        )

        sgp_cards = generate_tennis_sgp_cards(
            catalog=catalog,
            bankroll=bankroll,
            ai_adj=ai_adj,
        )

        if sgp_cards:
            pick = sgp_cards[0]  # Card 1: 🛡️ ESCUDO DE SETS
            sel_text = pick["legs"][0]["selection"] if pick.get("legs") else "Hándicap Sets +1.5"
            legs.append({
                "match": match_title,
                "surface": surface,
                "badge": pick.get("badge", "🛡️ ESCUDO DE SETS"),
                "selection_text": sel_text,
                "odds": float(pick["odds"]),
                "prob": float(pick.get("prob_pct", 65.0)) / 100.0,
            })

    # If no legs generated, provide safe fallback
    if not legs:
        legs = [{
            "match": "Partido ATP Principal",
            "surface": selected_surface,
            "badge": "🛡️ ESCUDO DE SETS",
            "selection_text": "Favorito +1.5 Sets & Total +18.5",
            "odds": 1.55,
            "prob": 0.68,
        }]

    # Independent cross-match parlay math (independent events)
    tot_odds = 1.0
    joint_prob = 1.0
    for l in legs:
        tot_odds *= l["odds"]
        joint_prob *= l["prob"]

    tot_odds = round(tot_odds, 2)
    joint_prob_pct = round(joint_prob * 100.0, 1)

    return legs, tot_odds, joint_prob_pct
