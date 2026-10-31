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
    cache_decorator = st.cache_data(ttl=300, show_spinner=False)
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


@cache_decorator
def fetch_tennis_odds(
    sport_key: str,
    api_key: Optional[str] = None,
    regions: str = "eu,us",
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
        return get_simulation_matches(sport_key), metadata, None

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
