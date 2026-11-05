"""Tennis Odds API Client & Realistic Match Simulator for QuantCourt AI.

Fetches live odds from The Odds API (v4) for ATP & WTA events,
and provides rich, surface-differentiated simulation datasets for testing without API keys.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests
try:
    import streamlit as st
    cache_decorator = st.cache_data(ttl=14400, show_spinner=False)
except Exception:
    def cache_decorator(func):
        return func

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

TENNIS_SPORTS: Dict[str, Dict[str, Any]] = {
    "tennis_atp_indian_wells": {
        "name": "Indian Wells Masters (ATP)",
        "surface": "Dura Outdoor",
        "type": "atp",
        "best_of": 3,
        "emoji": "🔵",
    },
    "tennis_atp_miami_open": {
        "name": "Miami Open (ATP)",
        "surface": "Dura Outdoor",
        "type": "atp",
        "best_of": 3,
        "emoji": "🔵",
    },
    "tennis_atp_rome": {
        "name": "Internazionali BNL d'Italia - Roma (ATP)",
        "surface": "Arcilla",
        "type": "atp",
        "best_of": 3,
        "emoji": "🟧",
    },
    "tennis_atp_madrid_open": {
        "name": "Mutua Madrid Open (ATP)",
        "surface": "Arcilla",
        "type": "atp",
        "best_of": 3,
        "emoji": "🟧",
    },
    "tennis_atp_french_open": {
        "name": "Roland Garros (ATP)",
        "surface": "Arcilla",
        "type": "atp",
        "best_of": 5,
        "emoji": "🟧",
    },
    "tennis_atp_wimbledon": {
        "name": "The Championships Wimbledon (ATP)",
        "surface": "Hierba",
        "type": "atp",
        "best_of": 5,
        "emoji": "🟩",
    },
    "tennis_atp_us_open": {
        "name": "US Open (ATP)",
        "surface": "Dura Outdoor",
        "type": "atp",
        "best_of": 5,
        "emoji": "🔵",
    },
    "tennis_atp_australian_open": {
        "name": "Australian Open (ATP)",
        "surface": "Dura Outdoor",
        "type": "atp",
        "best_of": 5,
        "emoji": "🔵",
    },
    "tennis_atp_paris": {
        "name": "Rolex Paris Masters (ATP)",
        "surface": "Dura Indoor",
        "type": "atp",
        "best_of": 3,
        "emoji": "🟦",
    },
    "tennis_atp_vienna": {
        "name": "Erste Bank Open - Viena (ATP)",
        "surface": "Dura Indoor",
        "type": "atp",
        "best_of": 3,
        "emoji": "🟦",
    },
    "tennis_wta_indian_wells": {
        "name": "BNP Paribas Open - Indian Wells (WTA)",
        "surface": "Dura Outdoor",
        "type": "wta",
        "best_of": 3,
        "emoji": "🔵",
    },
    "tennis_wta_rome": {
        "name": "Internazionali d'Italia (WTA)",
        "surface": "Arcilla",
        "type": "wta",
        "best_of": 3,
        "emoji": "🟧",
    },
    "tennis_wta_wimbledon": {
        "name": "Wimbledon Championships (WTA)",
        "surface": "Hierba",
        "type": "wta",
        "best_of": 3,
        "emoji": "🟩",
    },
}


def get_available_tennis_tournaments() -> Dict[str, Dict[str, Any]]:
    """Returns the catalog of configured tennis tournaments."""
    return TENNIS_SPORTS


def format_to_local_time(iso_str: str) -> str:
    """Formats an ISO-8601 UTC timestamp to localized human-readable string."""
    if not iso_str:
        return "Fecha por definir"
    try:
        clean_str = iso_str.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(clean_str)
        # Convert to local time (America/Bogota UTC-5 approx or local system)
        local_dt = dt_utc.astimezone()
        days_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        months_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        day_name = days_es[local_dt.weekday()]
        month_name = months_es[local_dt.month - 1]
        return f"{day_name} {local_dt.day} {month_name}, {local_dt.strftime('%H:%M')}"
    except Exception:
        return iso_str[:16].replace("T", " ")


def format_short_local_time(iso_str: str) -> str:
    """Returns a short time representation (e.g., '14:30' or 'Mañana 16:00')."""
    if not iso_str:
        return "TBD"
    try:
        clean_str = iso_str.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(clean_str)
        local_dt = dt_utc.astimezone()
        return local_dt.strftime("%d/%m %H:%M")
    except Exception:
        return iso_str[5:16].replace("T", " ")


def get_simulation_matches(sport_key: str) -> List[Dict[str, Any]]:
    """Generates 6+ high-realism tennis matches with differentiated surfaces,
    Pinnacle (oracle reference) and BetPlay odds with built-in value opportunities.
    """
    now = datetime.now(timezone.utc)
    t1 = (now + timedelta(days=1, hours=2)).isoformat()
    t2 = (now + timedelta(days=1, hours=4, minutes=30)).isoformat()
    t3 = (now + timedelta(days=1, hours=7)).isoformat()
    t4 = (now + timedelta(days=2, hours=1)).isoformat()
    t5 = (now + timedelta(days=2, hours=3, minutes=45)).isoformat()
    t6 = (now + timedelta(days=2, hours=6)).isoformat()

    tour_meta = TENNIS_SPORTS.get(sport_key, TENNIS_SPORTS["tennis_atp_indian_wells"])
    surface = tour_meta.get("surface", "Dura Outdoor")
    tour_title = tour_meta.get("name", "ATP Tour")

    # Matchups tailored by surface for realism
    if surface == "Arcilla":
        fixtures = [
            ("Carlos Alcaraz", "Casper Ruud", 1.34, 3.40, 1.40, 3.10, t1),
            ("Jannik Sinner", "Francisco Cerundolo", 1.25, 4.20, 1.30, 3.80, t2),
            ("Novak Djokovic", "Stefanos Tsitsipas", 1.58, 2.45, 1.68, 2.25, t3),
            ("Lorenzo Musetti", "Sebastian Baez", 1.76, 2.12, 1.88, 1.95, t4),
            ("Alexander Zverev", "Tomas Martin Etcheverry", 1.48, 2.75, 1.57, 2.50, t5),
            ("Andrey Rublev", "Alejandro Tabilo", 1.62, 2.35, 1.72, 2.15, t6),
        ]
    elif surface == "Hierba":
        fixtures = [
            ("Novak Djokovic", "Matteo Berrettini", 1.52, 2.60, 1.62, 2.38, t1),
            ("Carlos Alcaraz", "Hubert Hurkacz", 1.42, 2.95, 1.52, 2.65, t2),
            ("Jannik Sinner", "Jack Draper", 1.38, 3.15, 1.46, 2.85, t3),
            ("Alex De Minaur", "Ben Shelton", 1.82, 2.02, 1.95, 1.88, t4),
            ("Taylor Fritz", "Grigor Dimitrov", 1.68, 2.22, 1.78, 2.08, t5),
            ("Alexander Bublik", "Holger Rune", 1.92, 1.90, 2.05, 1.80, t6),
        ]
    elif surface == "Dura Indoor":
        fixtures = [
            ("Jannik Sinner", "Hubert Hurkacz", 1.28, 3.80, 1.35, 3.35, t1),
            ("Daniil Medvedev", "Holger Rune", 1.55, 2.50, 1.66, 2.30, t2),
            ("Alexander Zverev", "Arthur Fils", 1.44, 2.85, 1.52, 2.60, t3),
            ("Novak Djokovic", "Tomas Machac", 1.32, 3.55, 1.38, 3.20, t4),
            ("Jack Draper", "Ugo Humbert", 1.72, 2.18, 1.84, 2.00, t5),
            ("Grigor Dimitrov", "Jiri Lehecka", 1.65, 2.30, 1.75, 2.15, t6),
        ]
    elif tour_meta.get("type") == "wta":
        fixtures = [
            ("Iga Swiatek", "Jessica Pegula", 1.28, 3.75, 1.35, 3.30, t1),
            ("Aryna Sabalenka", "Coco Gauff", 1.62, 2.35, 1.72, 2.18, t2),
            ("Elena Rybakina", "Zheng Qinwen", 1.52, 2.58, 1.62, 2.36, t3),
            ("Jasmine Paolini", "Daria Kasatkina", 1.75, 2.12, 1.86, 1.98, t4),
            ("Madison Keys", "Emma Navarro", 1.82, 2.02, 1.95, 1.90, t5),
            ("Paula Badosa", "Donna Vekic", 1.65, 2.30, 1.74, 2.16, t6),
        ]
    else:  # Dura Outdoor
        fixtures = [
            ("Jannik Sinner", "Andrey Rublev", 1.24, 4.30, 1.30, 3.75, t1),
            ("Carlos Alcaraz", "Alexander Zverev", 1.54, 2.55, 1.64, 2.32, t2),
            ("Daniil Medvedev", "Taylor Fritz", 1.68, 2.22, 1.78, 2.10, t3),
            ("Novak Djokovic", "Ben Shelton", 1.36, 3.25, 1.44, 2.90, t4),
            ("Stefanos Tsitsipas", "Alex De Minaur", 1.85, 1.98, 1.96, 1.88, t5),
            ("Casper Ruud", "Hubert Hurkacz", 1.92, 1.90, 2.05, 1.82, t6),
        ]

    events: List[Dict[str, Any]] = []
    for idx, (p1, p2, pin_1, pin_2, bp_1, bp_2, t_comm) in enumerate(fixtures):
        events.append({
            "id": f"sim_match_{sport_key}_{idx+1}",
            "sport_key": sport_key,
            "sport_title": tour_title,
            "surface": surface,
            "commence_time": t_comm,
            "home_team": p1,
            "away_team": p2,
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle (Oráculo)",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": p1, "price": pin_1},
                                {"name": p2, "price": pin_2},
                            ],
                        }
                    ],
                },
                {
                    "key": "betplay",
                    "title": "BetPlay",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": p1, "price": bp_1},
                                {"name": p2, "price": bp_2},
                            ],
                        }
                    ],
                },
                {
                    "key": "wplay",
                    "title": "Wplay",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": p1, "price": round(bp_1 * 0.99, 2)},
                                {"name": p2, "price": round(bp_2 * 1.01, 2)},
                            ],
                        }
                    ],
                },
            ],
        })

    return events


SHARP_TENNIS_PRIORITY: Dict[str, int] = {
    "pinnacle": 1,
    "betfair_ex_eu": 2,
    "betfair_sb_uk": 3,
    "betfair": 4,
    "bet365": 5,
    "williamhill": 6,
    "unibet": 7,
    "betsson": 8,
    "betplay": 9,
    "rushbet": 10,
    "wplay": 11,
}


def calc_tennis_vig_free_probs(o1: float, o2: float) -> Tuple[float, float]:
    """Calculates clean Vig-Free probabilities for a 2-way tennis market."""
    if o1 <= 1.0 or o2 <= 1.0:
        return 0.50, 0.50
    inv1 = 1.0 / o1
    inv2 = 1.0 / o2
    tot = inv1 + inv2
    return round(inv1 / tot, 4), round(inv2 / tot, 4)


def extract_event_tennis_odds(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts the highest-liquidity sharp market odds for Player 1 and Player 2 from an event.

    Prioritizes global sharp books (Pinnacle, Betfair, Bet365) and falls back to
    available retail books. Returns clean Vig-Free probabilities and odds dictionary.
    """
    p1 = event.get("home_team", "Jugador A")
    p2 = event.get("away_team", "Jugador B")

    best_bm: Optional[Dict[str, Any]] = None
    best_rank = 999

    for bm in event.get("bookmakers", []):
        bm_key = str(bm.get("key", "")).lower()
        rank = SHARP_TENNIS_PRIORITY.get(bm_key, 50)
        has_h2h = any(m.get("key") == "h2h" for m in bm.get("markets", []))
        if has_h2h and rank < best_rank:
            best_rank = rank
            best_bm = bm

    o1 = 1.85
    o2 = 1.85
    bm_name = "Sharp Book"
    bm_key = "pinnacle"

    if best_bm:
        bm_name = best_bm.get("title", best_bm.get("key", "Sharp Book"))
        bm_key = best_bm.get("key", "sharp")
        for m in best_bm.get("markets", []):
            if m.get("key") == "h2h":
                for o in m.get("outcomes", []):
                    if o.get("name") == p1:
                        o1 = float(o.get("price", o1))
                    elif o.get("name") == p2:
                        o2 = float(o.get("price", o2))

    vf1, vf2 = calc_tennis_vig_free_probs(o1, o2)

    return {
        "1": o1,
        "2": o2,
        "vig_free_1": vf1,
        "vig_free_2": vf2,
        "bookmaker": bm_name,
        "bookmaker_key": bm_key,
        "is_sharp": best_rank <= 7,
    }



# ==============================================================================
# SMART MONEY & STEAM CHASING IN TENNIS (LINE DROP TRACKER)
# ==============================================================================

_TENNIS_STEAM_CACHE: Dict[str, Dict[str, Any]] = {}

def extract_sharp_tennis_odds(event: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], str]:
    """Pulls current Pinnacle or Sharp consensus odds for Player 1 and Player 2."""
    p1 = event.get("home_team", "Player 1")
    p2 = event.get("away_team", "Player 2")
    bookmakers = event.get("bookmakers", [])

    pinnacle = next((b for b in bookmakers if b.get("key") == "pinnacle"), None)
    target_bm = pinnacle if pinnacle else (bookmakers[0] if bookmakers else None)

    if not target_bm:
        return None, "None"

    bm_name = target_bm.get("title", target_bm.get("key", "Sharp"))
    odds_map: Dict[str, float] = {}
    for m in target_bm.get("markets", []):
        if m.get("key") == "h2h":
            for o in m.get("outcomes", []):
                if o.get("name") == p1:
                    odds_map["1"] = float(o.get("price", 0.0))
                elif o.get("name") == p2:
                    odds_map["2"] = float(o.get("price", 0.0))
            break

    if "1" in odds_map and "2" in odds_map:
        return odds_map, bm_name
    return None, bm_name


def update_tennis_steam_cache(event: Dict[str, Any]) -> Dict[str, Any]:
    """Caches opening line and updates latest odds to detect line movement in tennis."""
    p1 = event.get("home_team", "")
    p2 = event.get("away_team", "")
    ev_id = str(event.get("id") or f"{p1} vs {p2}")
    now = datetime.now()
    curr_odds, bm_name = extract_sharp_tennis_odds(event)

    sim_open = event.get("opening_odds")
    sim_time_ago = float(event.get("opening_hours_ago", 2.5))

    if ev_id not in _TENNIS_STEAM_CACHE:
        if sim_open:
            _TENNIS_STEAM_CACHE[ev_id] = {
                "first_seen": now - timedelta(hours=sim_time_ago),
                "open_odds": dict(sim_open),
                "last_seen": now,
                "current_odds": curr_odds if curr_odds else dict(sim_open),
                "bookmaker": bm_name,
                "p1": p1,
                "p2": p2,
            }
        else:
            if curr_odds:
                _TENNIS_STEAM_CACHE[ev_id] = {
                    "first_seen": now,
                    "open_odds": dict(curr_odds),
                    "last_seen": now,
                    "current_odds": dict(curr_odds),
                    "bookmaker": bm_name,
                    "p1": p1,
                    "p2": p2,
                }
    else:
        cached = _TENNIS_STEAM_CACHE[ev_id]
        cached["last_seen"] = now
        if curr_odds:
            cached["current_odds"] = dict(curr_odds)
            cached["bookmaker"] = bm_name

    return _TENNIS_STEAM_CACHE.get(ev_id, {})


def detect_tennis_steam_moves(
    event_or_odds: Any,
    window_hours: float = 4.0,
    threshold_pct: float = 10.0,
) -> Dict[str, Any]:
    """Detects if a tennis player's odds dropped > 10% in the last 4 hours (Steam Chasing).
    In professional tennis, this strongly correlates with opponent injury / illness alerts
    or institutional sharp syndicate money.
    """
    if isinstance(event_or_odds, dict) and ("home_team" in event_or_odds or "bookmakers" in event_or_odds):
        event = event_or_odds
        cached = update_tennis_steam_cache(event)
    elif isinstance(event_or_odds, str) and event_or_odds in _TENNIS_STEAM_CACHE:
        cached = _TENNIS_STEAM_CACHE[event_or_odds]
    elif isinstance(event_or_odds, dict) and "open_odds" in event_or_odds:
        cached = event_or_odds
    else:
        cached = {}

    if not cached or "open_odds" not in cached or "current_odds" not in cached:
        return {
            "is_steam": False,
            "steam_detected": False,
            "drop_pct": 0.0,
            "side": None,
            "player": "",
            "open_odds": 0.0,
            "current_odds": 0.0,
            "hours_elapsed": 0.0,
            "badge": "⚪ LÍNEA ESTABLE",
            "description": "Sin datos de apertura suficientes.",
            "bonus_points": 0.0,
        }

    first_seen = cached["first_seen"]
    last_seen = cached["last_seen"]
    hours_elapsed = max(0.1, (last_seen - first_seen).total_seconds() / 3600.0)

    open_o = cached["open_odds"]
    curr_o = cached["current_odds"]
    p1 = cached.get("p1", "Jugador 1")
    p2 = cached.get("p2", "Jugador 2")

    max_drop = 0.0
    favored_side = None
    favored_player = ""
    open_val = 0.0
    curr_val = 0.0

    for side, p_name in [("1", p1), ("2", p2)]:
        if side in open_o and side in curr_o:
            o_init = float(open_o[side])
            o_now = float(curr_o[side])
            if o_init > 1.0 and o_now < o_init:
                drop_pct = ((o_init - o_now) / o_init) * 100.0
                if drop_pct > max_drop:
                    max_drop = drop_pct
                    favored_side = side
                    favored_player = p_name
                    open_val = o_init
                    curr_val = o_now

    is_steam = (max_drop >= threshold_pct) and (hours_elapsed <= window_hours)

    if is_steam:
        badge = "🚨 ALERTA SMART MONEY (Alerta Lesión / Dinero Institucional)"
        desc = (
            f"Caída de cuota institucional de {max_drop:.1f}% en {favored_player} "
            f"(@{open_val:.2f} ➔ @{curr_val:.2f}) en las últimas {hours_elapsed:.1f}h. "
            f"Posible alerta de lesión oculta del rival o flujo masivo en Pinnacle."
        )
        bonus = min(100.0, max(20.0, max_drop * 5.0))
    else:
        badge = "⚪ LÍNEA ESTABLE"
        desc = "Fluctuación regular del mercado sin movimientos institucionales abruptos."
        bonus = 0.0

    return {
        "is_steam": is_steam,
        "steam_detected": is_steam,
        "drop_pct": round(max_drop, 1),
        "side": favored_side,
        "player": favored_player,
        "open_odds": open_val,
        "current_odds": curr_val,
        "hours_elapsed": round(hours_elapsed, 1),
        "bookmaker": cached.get("bookmaker", "Pinnacle"),
        "badge": badge,
        "description": desc,
        "bonus_points": round(bonus, 1),
    }


@cache_decorator
def fetch_tennis_odds(
    sport_key: str,
    api_key: Optional[str] = None,
    regions: str = "eu,uk,us",
    markets: str = "h2h",
    is_demo: bool = True,
    force_nonce: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """Fetches real-time odds from The Odds API or provides realistic simulation data.

    Returns:
        (events, metadata, error_message)
    """
    metadata: Dict[str, Any] = {
        "requests_remaining": "N/A",
        "requests_used": "N/A",
        "source": "demo" if is_demo else "api",
        "sport_key": sport_key,
    }

    if is_demo or not api_key or not api_key.strip():
        metadata["requests_remaining"] = "500 (Modo Demo Activo)"
        metadata["requests_used"] = "0"
        metadata["source"] = "demo"
        matches = get_simulation_matches(sport_key)
        for m in matches:
            update_tennis_steam_cache(m)
            m["steam_move"] = detect_tennis_steam_moves(m)
        return matches, metadata, None

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key.strip(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        try:
            resp = requests.get(url, params=params, timeout=12)
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.get(url, params=params, timeout=12, verify=False)

        if "x-requests-remaining" in resp.headers:
            metadata["requests_remaining"] = resp.headers.get("x-requests-remaining")
        if "x-requests-used" in resp.headers:
            metadata["requests_used"] = resp.headers.get("x-requests-used")

        if resp.status_code == 200:
            data = resp.json()
            if not data:
                return (
                    get_simulation_matches(sport_key),
                    metadata,
                    f"No hay partidos activos en vivo en The Odds API para {sport_key}. Cargando partidos de demostración.",
                )
            # Tag surface onto events
            tour_meta = TENNIS_SPORTS.get(sport_key, {})
            surface = tour_meta.get("surface", "Dura Outdoor")
            for ev in data:
                ev["surface"] = surface
                update_tennis_steam_cache(ev)
                ev["steam_move"] = detect_tennis_steam_moves(ev)
            return data, metadata, None

        if resp.status_code == 401:
            return (
                get_simulation_matches(sport_key),
                metadata,
                "API Key no válida o expirada. Mostrando partidos en modo demostración.",
            )
        elif resp.status_code == 429:
            return (
                get_simulation_matches(sport_key),
                metadata,
                "Cuota mensual de The Odds API alcanzada (500/500). Mostrando modo demostración.",
            )
        else:
            return (
                get_simulation_matches(sport_key),
                metadata,
                f"Error {resp.status_code} desde The Odds API. Mostrando modo demostración.",
            )

    except Exception as e:
        return (
            get_simulation_matches(sport_key),
            metadata,
            f"Error de conexión ({type(e).__name__}). Mostrando partidos de demostración.",
        )
