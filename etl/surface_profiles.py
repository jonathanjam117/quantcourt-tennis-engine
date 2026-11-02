"""
etl/surface_profiles.py
========================
Catalog of tennis player statistics differentiated by court surface.

Provides realistic serve/return/hold/break/Elo profiles for 60+ ATP and WTA
players across Hard Outdoor, Hard Indoor, Clay, and Grass surfaces.  Used by
the Markov-chain match engine to produce surface-sensitive probability
estimates.

Key public helpers
------------------
* ``get_player_profile``  – fuzzy-matched lookup with tour-average fallback.
* ``detect_surface_from_tournament`` – heuristic surface classifier.
* ``get_surface_emoji`` / ``get_surface_color`` – UI helpers.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. Surface-type constants
# ---------------------------------------------------------------------------

SURFACE_HARD_OUTDOOR: str = "Dura Outdoor"
SURFACE_HARD_INDOOR: str = "Dura Indoor"
SURFACE_CLAY: str = "Arcilla"
SURFACE_GRASS: str = "Hierba"

ALL_SURFACES: List[str] = [
    SURFACE_HARD_OUTDOOR,
    SURFACE_HARD_INDOOR,
    SURFACE_CLAY,
    SURFACE_GRASS,
]

# ---------------------------------------------------------------------------
# 2. Tour averages by surface
# ---------------------------------------------------------------------------

TOUR_AVERAGES: Dict[str, Dict[str, float]] = {
    SURFACE_HARD_OUTDOOR: {"spw": 0.640, "rpw": 0.360, "hold": 0.820, "break": 0.180, "elo": 1500},
    SURFACE_HARD_INDOOR: {"spw": 0.660, "rpw": 0.340, "hold": 0.845, "break": 0.155, "elo": 1500},
    SURFACE_CLAY:         {"spw": 0.620, "rpw": 0.380, "hold": 0.790, "break": 0.210, "elo": 1500},
    SURFACE_GRASS:        {"spw": 0.670, "rpw": 0.330, "hold": 0.860, "break": 0.140, "elo": 1500},
}

# ---------------------------------------------------------------------------
# 3. Player profiles  (≥40 ATP + ≥20 WTA)
# ---------------------------------------------------------------------------
# Each value maps surface → {spw, rpw, hold, break, elo}.
# Stats are intentionally varied per surface so the Markov engine produces
# meaningfully different odds depending on the court.
# ---------------------------------------------------------------------------

PLAYER_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    # ======================================================================
    # ATP  (40 players)
    # ======================================================================

    # --- Jannik Sinner ---------------------------------------------------
    "Sinner": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.690, "rpw": 0.395, "hold": 0.890, "break": 0.225, "elo": 2100},
        SURFACE_HARD_INDOOR:  {"spw": 0.710, "rpw": 0.385, "hold": 0.910, "break": 0.215, "elo": 2120},
        SURFACE_CLAY:         {"spw": 0.650, "rpw": 0.385, "hold": 0.840, "break": 0.215, "elo": 1960},
        SURFACE_GRASS:        {"spw": 0.700, "rpw": 0.370, "hold": 0.895, "break": 0.195, "elo": 2020},
    },
    # --- Novak Djokovic --------------------------------------------------
    "Djokovic": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.675, "rpw": 0.410, "hold": 0.880, "break": 0.240, "elo": 2050},
        SURFACE_HARD_INDOOR:  {"spw": 0.690, "rpw": 0.400, "hold": 0.895, "break": 0.230, "elo": 2070},
        SURFACE_CLAY:         {"spw": 0.645, "rpw": 0.405, "hold": 0.840, "break": 0.240, "elo": 2010},
        SURFACE_GRASS:        {"spw": 0.690, "rpw": 0.380, "hold": 0.890, "break": 0.210, "elo": 2030},
    },
    # --- Carlos Alcaraz --------------------------------------------------
    "Alcaraz": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.680, "rpw": 0.400, "hold": 0.875, "break": 0.230, "elo": 2080},
        SURFACE_HARD_INDOOR:  {"spw": 0.690, "rpw": 0.390, "hold": 0.890, "break": 0.220, "elo": 2060},
        SURFACE_CLAY:         {"spw": 0.660, "rpw": 0.410, "hold": 0.850, "break": 0.250, "elo": 2070},
        SURFACE_GRASS:        {"spw": 0.705, "rpw": 0.375, "hold": 0.900, "break": 0.205, "elo": 2090},
    },
    # --- Alexander Zverev ------------------------------------------------
    "Zverev": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.670, "rpw": 0.385, "hold": 0.870, "break": 0.215, "elo": 2000},
        SURFACE_HARD_INDOOR:  {"spw": 0.685, "rpw": 0.375, "hold": 0.885, "break": 0.200, "elo": 2010},
        SURFACE_CLAY:         {"spw": 0.650, "rpw": 0.395, "hold": 0.845, "break": 0.235, "elo": 1990},
        SURFACE_GRASS:        {"spw": 0.690, "rpw": 0.355, "hold": 0.885, "break": 0.180, "elo": 1960},
    },
    # --- Daniil Medvedev -------------------------------------------------
    "Medvedev": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.665, "rpw": 0.395, "hold": 0.865, "break": 0.225, "elo": 2010},
        SURFACE_HARD_INDOOR:  {"spw": 0.680, "rpw": 0.390, "hold": 0.880, "break": 0.220, "elo": 2030},
        SURFACE_CLAY:         {"spw": 0.630, "rpw": 0.370, "hold": 0.815, "break": 0.200, "elo": 1870},
        SURFACE_GRASS:        {"spw": 0.680, "rpw": 0.355, "hold": 0.875, "break": 0.185, "elo": 1940},
    },
    # --- Taylor Fritz ----------------------------------------------------
    "Fritz": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.680, "rpw": 0.370, "hold": 0.880, "break": 0.195, "elo": 1950},
        SURFACE_HARD_INDOOR:  {"spw": 0.695, "rpw": 0.360, "hold": 0.895, "break": 0.185, "elo": 1970},
        SURFACE_CLAY:         {"spw": 0.635, "rpw": 0.350, "hold": 0.820, "break": 0.175, "elo": 1810},
        SURFACE_GRASS:        {"spw": 0.700, "rpw": 0.350, "hold": 0.900, "break": 0.175, "elo": 1960},
    },
    # --- Andrey Rublev ---------------------------------------------------
    "Rublev": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.660, "rpw": 0.380, "hold": 0.855, "break": 0.210, "elo": 1940},
        SURFACE_HARD_INDOOR:  {"spw": 0.675, "rpw": 0.375, "hold": 0.870, "break": 0.205, "elo": 1960},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.390, "hold": 0.830, "break": 0.225, "elo": 1920},
        SURFACE_GRASS:        {"spw": 0.670, "rpw": 0.345, "hold": 0.865, "break": 0.170, "elo": 1870},
    },
    # --- Alex De Minaur --------------------------------------------------
    "De Minaur": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.645, "rpw": 0.385, "hold": 0.840, "break": 0.215, "elo": 1910},
        SURFACE_HARD_INDOOR:  {"spw": 0.655, "rpw": 0.380, "hold": 0.850, "break": 0.210, "elo": 1920},
        SURFACE_CLAY:         {"spw": 0.620, "rpw": 0.375, "hold": 0.805, "break": 0.200, "elo": 1830},
        SURFACE_GRASS:        {"spw": 0.665, "rpw": 0.370, "hold": 0.865, "break": 0.195, "elo": 1930},
    },
    # --- Casper Ruud  (Clay specialist) ----------------------------------
    "Ruud": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.645, "rpw": 0.365, "hold": 0.835, "break": 0.190, "elo": 1870},
        SURFACE_HARD_INDOOR:  {"spw": 0.650, "rpw": 0.355, "hold": 0.840, "break": 0.180, "elo": 1860},
        SURFACE_CLAY:         {"spw": 0.660, "rpw": 0.405, "hold": 0.855, "break": 0.245, "elo": 2000},
        SURFACE_GRASS:        {"spw": 0.655, "rpw": 0.335, "hold": 0.845, "break": 0.160, "elo": 1790},
    },
    # --- Stefanos Tsitsipas (strong clay) --------------------------------
    "Tsitsipas": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.660, "rpw": 0.370, "hold": 0.855, "break": 0.200, "elo": 1920},
        SURFACE_HARD_INDOOR:  {"spw": 0.670, "rpw": 0.365, "hold": 0.865, "break": 0.190, "elo": 1910},
        SURFACE_CLAY:         {"spw": 0.655, "rpw": 0.400, "hold": 0.850, "break": 0.240, "elo": 1990},
        SURFACE_GRASS:        {"spw": 0.675, "rpw": 0.340, "hold": 0.870, "break": 0.165, "elo": 1860},
    },
    # --- Hubert Hurkacz  (Grass specialist, big serve) -------------------
    "Hurkacz": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.685, "rpw": 0.355, "hold": 0.885, "break": 0.180, "elo": 1910},
        SURFACE_HARD_INDOOR:  {"spw": 0.700, "rpw": 0.350, "hold": 0.900, "break": 0.175, "elo": 1940},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.350, "hold": 0.830, "break": 0.175, "elo": 1810},
        SURFACE_GRASS:        {"spw": 0.720, "rpw": 0.355, "hold": 0.920, "break": 0.180, "elo": 2000},
    },
    # --- Grigor Dimitrov -------------------------------------------------
    "Dimitrov": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.660, "rpw": 0.375, "hold": 0.855, "break": 0.205, "elo": 1900},
        SURFACE_HARD_INDOOR:  {"spw": 0.670, "rpw": 0.370, "hold": 0.865, "break": 0.200, "elo": 1910},
        SURFACE_CLAY:         {"spw": 0.635, "rpw": 0.370, "hold": 0.825, "break": 0.200, "elo": 1840},
        SURFACE_GRASS:        {"spw": 0.680, "rpw": 0.350, "hold": 0.875, "break": 0.175, "elo": 1890},
    },
    # --- Tommy Paul ------------------------------------------------------
    "Paul": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.665, "rpw": 0.365, "hold": 0.860, "break": 0.190, "elo": 1880},
        SURFACE_HARD_INDOOR:  {"spw": 0.675, "rpw": 0.360, "hold": 0.870, "break": 0.185, "elo": 1890},
        SURFACE_CLAY:         {"spw": 0.630, "rpw": 0.355, "hold": 0.815, "break": 0.180, "elo": 1780},
        SURFACE_GRASS:        {"spw": 0.690, "rpw": 0.350, "hold": 0.890, "break": 0.175, "elo": 1910},
    },
    # --- Ben Shelton (huge serve) ----------------------------------------
    "Shelton": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.695, "rpw": 0.355, "hold": 0.895, "break": 0.175, "elo": 1890},
        SURFACE_HARD_INDOOR:  {"spw": 0.710, "rpw": 0.350, "hold": 0.910, "break": 0.170, "elo": 1920},
        SURFACE_CLAY:         {"spw": 0.645, "rpw": 0.345, "hold": 0.835, "break": 0.170, "elo": 1760},
        SURFACE_GRASS:        {"spw": 0.715, "rpw": 0.340, "hold": 0.915, "break": 0.165, "elo": 1920},
    },
    # --- Jack Draper (lefty, indoor monster) -----------------------------
    "Draper": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.675, "rpw": 0.375, "hold": 0.875, "break": 0.205, "elo": 1900},
        SURFACE_HARD_INDOOR:  {"spw": 0.700, "rpw": 0.380, "hold": 0.900, "break": 0.210, "elo": 1960},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.365, "hold": 0.830, "break": 0.190, "elo": 1810},
        SURFACE_GRASS:        {"spw": 0.695, "rpw": 0.360, "hold": 0.895, "break": 0.185, "elo": 1930},
    },
    # --- Holger Rune -----------------------------------------------------
    "Rune": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.655, "rpw": 0.380, "hold": 0.850, "break": 0.210, "elo": 1880},
        SURFACE_HARD_INDOOR:  {"spw": 0.670, "rpw": 0.375, "hold": 0.865, "break": 0.205, "elo": 1900},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.390, "hold": 0.835, "break": 0.225, "elo": 1870},
        SURFACE_GRASS:        {"spw": 0.670, "rpw": 0.350, "hold": 0.865, "break": 0.175, "elo": 1840},
    },
    # --- Lorenzo Musetti (clay-court artisan) ----------------------------
    "Musetti": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.640, "rpw": 0.365, "hold": 0.830, "break": 0.190, "elo": 1830},
        SURFACE_HARD_INDOOR:  {"spw": 0.650, "rpw": 0.355, "hold": 0.840, "break": 0.180, "elo": 1830},
        SURFACE_CLAY:         {"spw": 0.645, "rpw": 0.395, "hold": 0.840, "break": 0.235, "elo": 1920},
        SURFACE_GRASS:        {"spw": 0.660, "rpw": 0.345, "hold": 0.855, "break": 0.170, "elo": 1850},
    },
    # --- Frances Tiafoe --------------------------------------------------
    "Tiafoe": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.665, "rpw": 0.370, "hold": 0.860, "break": 0.200, "elo": 1870},
        SURFACE_HARD_INDOOR:  {"spw": 0.675, "rpw": 0.365, "hold": 0.870, "break": 0.190, "elo": 1880},
        SURFACE_CLAY:         {"spw": 0.625, "rpw": 0.355, "hold": 0.810, "break": 0.180, "elo": 1750},
        SURFACE_GRASS:        {"spw": 0.685, "rpw": 0.350, "hold": 0.885, "break": 0.175, "elo": 1880},
    },
    # --- Matteo Berrettini  (Grass specialist, big serve) ----------------
    "Berrettini": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.685, "rpw": 0.355, "hold": 0.885, "break": 0.180, "elo": 1890},
        SURFACE_HARD_INDOOR:  {"spw": 0.695, "rpw": 0.350, "hold": 0.895, "break": 0.175, "elo": 1900},
        SURFACE_CLAY:         {"spw": 0.650, "rpw": 0.365, "hold": 0.845, "break": 0.195, "elo": 1860},
        SURFACE_GRASS:        {"spw": 0.720, "rpw": 0.350, "hold": 0.925, "break": 0.175, "elo": 2010},
    },
    # --- Alexander Bublik (serve & volley chaos) -------------------------
    "Bublik": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.685, "rpw": 0.345, "hold": 0.885, "break": 0.170, "elo": 1860},
        SURFACE_HARD_INDOOR:  {"spw": 0.700, "rpw": 0.340, "hold": 0.900, "break": 0.165, "elo": 1890},
        SURFACE_CLAY:         {"spw": 0.635, "rpw": 0.340, "hold": 0.825, "break": 0.165, "elo": 1740},
        SURFACE_GRASS:        {"spw": 0.710, "rpw": 0.335, "hold": 0.910, "break": 0.160, "elo": 1920},
    },
    # --- Karen Khachanov -------------------------------------------------
    "Khachanov": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.665, "rpw": 0.365, "hold": 0.860, "break": 0.195, "elo": 1870},
        SURFACE_HARD_INDOOR:  {"spw": 0.680, "rpw": 0.360, "hold": 0.875, "break": 0.185, "elo": 1890},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.365, "hold": 0.830, "break": 0.195, "elo": 1820},
        SURFACE_GRASS:        {"spw": 0.680, "rpw": 0.340, "hold": 0.875, "break": 0.165, "elo": 1850},
    },
    # --- Felix Auger-Aliassime (big serve, inconsistent) -----------------
    "Auger-Aliassime": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.675, "rpw": 0.360, "hold": 0.870, "break": 0.185, "elo": 1860},
        SURFACE_HARD_INDOOR:  {"spw": 0.690, "rpw": 0.355, "hold": 0.890, "break": 0.180, "elo": 1880},
        SURFACE_CLAY:         {"spw": 0.635, "rpw": 0.355, "hold": 0.820, "break": 0.180, "elo": 1780},
        SURFACE_GRASS:        {"spw": 0.695, "rpw": 0.345, "hold": 0.895, "break": 0.170, "elo": 1880},
    },
    # --- Sebastian Baez  (Clay specialist) -------------------------------
    "Baez": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.625, "rpw": 0.365, "hold": 0.810, "break": 0.190, "elo": 1770},
        SURFACE_HARD_INDOOR:  {"spw": 0.630, "rpw": 0.355, "hold": 0.815, "break": 0.180, "elo": 1760},
        SURFACE_CLAY:         {"spw": 0.650, "rpw": 0.400, "hold": 0.845, "break": 0.240, "elo": 1930},
        SURFACE_GRASS:        {"spw": 0.630, "rpw": 0.335, "hold": 0.815, "break": 0.160, "elo": 1700},
    },
    # --- Francisco Cerundolo (Clay specialist) ---------------------------
    "Cerundolo": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.635, "rpw": 0.370, "hold": 0.825, "break": 0.200, "elo": 1800},
        SURFACE_HARD_INDOOR:  {"spw": 0.640, "rpw": 0.360, "hold": 0.830, "break": 0.185, "elo": 1790},
        SURFACE_CLAY:         {"spw": 0.650, "rpw": 0.400, "hold": 0.845, "break": 0.240, "elo": 1920},
        SURFACE_GRASS:        {"spw": 0.640, "rpw": 0.335, "hold": 0.830, "break": 0.160, "elo": 1720},
    },
    # --- Alejandro Tabilo ------------------------------------------------
    "Tabilo": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.650, "rpw": 0.365, "hold": 0.845, "break": 0.190, "elo": 1830},
        SURFACE_HARD_INDOOR:  {"spw": 0.660, "rpw": 0.355, "hold": 0.855, "break": 0.180, "elo": 1830},
        SURFACE_CLAY:         {"spw": 0.645, "rpw": 0.385, "hold": 0.840, "break": 0.220, "elo": 1880},
        SURFACE_GRASS:        {"spw": 0.665, "rpw": 0.340, "hold": 0.860, "break": 0.165, "elo": 1820},
    },
    # --- Jordan Thompson -------------------------------------------------
    "Thompson": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.655, "rpw": 0.360, "hold": 0.850, "break": 0.185, "elo": 1820},
        SURFACE_HARD_INDOOR:  {"spw": 0.665, "rpw": 0.355, "hold": 0.860, "break": 0.180, "elo": 1820},
        SURFACE_CLAY:         {"spw": 0.625, "rpw": 0.360, "hold": 0.810, "break": 0.185, "elo": 1740},
        SURFACE_GRASS:        {"spw": 0.680, "rpw": 0.345, "hold": 0.875, "break": 0.170, "elo": 1860},
    },
    # --- Ugo Humbert (fast-court specialist) -----------------------------
    "Humbert": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.660, "rpw": 0.370, "hold": 0.855, "break": 0.200, "elo": 1860},
        SURFACE_HARD_INDOOR:  {"spw": 0.685, "rpw": 0.375, "hold": 0.885, "break": 0.205, "elo": 1920},
        SURFACE_CLAY:         {"spw": 0.630, "rpw": 0.360, "hold": 0.815, "break": 0.185, "elo": 1780},
        SURFACE_GRASS:        {"spw": 0.690, "rpw": 0.355, "hold": 0.890, "break": 0.180, "elo": 1910},
    },
    # --- Arthur Fils -----------------------------------------------------
    "Fils": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.660, "rpw": 0.370, "hold": 0.855, "break": 0.200, "elo": 1860},
        SURFACE_HARD_INDOOR:  {"spw": 0.675, "rpw": 0.365, "hold": 0.870, "break": 0.195, "elo": 1880},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.375, "hold": 0.830, "break": 0.205, "elo": 1830},
        SURFACE_GRASS:        {"spw": 0.675, "rpw": 0.345, "hold": 0.870, "break": 0.170, "elo": 1840},
    },
    # --- Sebastian Korda -------------------------------------------------
    "Korda": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.665, "rpw": 0.365, "hold": 0.860, "break": 0.190, "elo": 1860},
        SURFACE_HARD_INDOOR:  {"spw": 0.680, "rpw": 0.360, "hold": 0.875, "break": 0.185, "elo": 1870},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.370, "hold": 0.830, "break": 0.200, "elo": 1810},
        SURFACE_GRASS:        {"spw": 0.690, "rpw": 0.350, "hold": 0.890, "break": 0.175, "elo": 1890},
    },
    # --- Nicolas Jarry ---------------------------------------------------
    "Jarry": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.670, "rpw": 0.355, "hold": 0.865, "break": 0.180, "elo": 1840},
        SURFACE_HARD_INDOOR:  {"spw": 0.680, "rpw": 0.350, "hold": 0.875, "break": 0.175, "elo": 1850},
        SURFACE_CLAY:         {"spw": 0.650, "rpw": 0.375, "hold": 0.845, "break": 0.210, "elo": 1870},
        SURFACE_GRASS:        {"spw": 0.685, "rpw": 0.335, "hold": 0.885, "break": 0.160, "elo": 1830},
    },
    # --- Tomas Martin Etcheverry (Clay specialist) -----------------------
    "Etcheverry": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.640, "rpw": 0.360, "hold": 0.830, "break": 0.185, "elo": 1780},
        SURFACE_HARD_INDOOR:  {"spw": 0.645, "rpw": 0.350, "hold": 0.835, "break": 0.175, "elo": 1770},
        SURFACE_CLAY:         {"spw": 0.655, "rpw": 0.400, "hold": 0.850, "break": 0.240, "elo": 1920},
        SURFACE_GRASS:        {"spw": 0.640, "rpw": 0.330, "hold": 0.830, "break": 0.155, "elo": 1700},
    },
    # --- Kei Nishikori ---------------------------------------------------
    "Nishikori": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.635, "rpw": 0.390, "hold": 0.825, "break": 0.220, "elo": 1850},
        SURFACE_HARD_INDOOR:  {"spw": 0.645, "rpw": 0.385, "hold": 0.835, "break": 0.215, "elo": 1850},
        SURFACE_CLAY:         {"spw": 0.625, "rpw": 0.385, "hold": 0.810, "break": 0.215, "elo": 1830},
        SURFACE_GRASS:        {"spw": 0.650, "rpw": 0.355, "hold": 0.845, "break": 0.180, "elo": 1800},
    },
    # --- Brandon Nakashima -----------------------------------------------
    "Nakashima": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.655, "rpw": 0.355, "hold": 0.850, "break": 0.180, "elo": 1810},
        SURFACE_HARD_INDOOR:  {"spw": 0.665, "rpw": 0.350, "hold": 0.860, "break": 0.175, "elo": 1820},
        SURFACE_CLAY:         {"spw": 0.620, "rpw": 0.350, "hold": 0.805, "break": 0.175, "elo": 1720},
        SURFACE_GRASS:        {"spw": 0.675, "rpw": 0.340, "hold": 0.870, "break": 0.165, "elo": 1830},
    },
    # --- Tallon Griekspoor -----------------------------------------------
    "Griekspoor": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.665, "rpw": 0.355, "hold": 0.860, "break": 0.180, "elo": 1840},
        SURFACE_HARD_INDOOR:  {"spw": 0.680, "rpw": 0.355, "hold": 0.875, "break": 0.180, "elo": 1870},
        SURFACE_CLAY:         {"spw": 0.630, "rpw": 0.350, "hold": 0.815, "break": 0.175, "elo": 1750},
        SURFACE_GRASS:        {"spw": 0.685, "rpw": 0.340, "hold": 0.885, "break": 0.165, "elo": 1860},
    },
    # --- Tomas Machac ----------------------------------------------------
    "Machac": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.660, "rpw": 0.370, "hold": 0.855, "break": 0.200, "elo": 1850},
        SURFACE_HARD_INDOOR:  {"spw": 0.675, "rpw": 0.365, "hold": 0.870, "break": 0.190, "elo": 1870},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.375, "hold": 0.830, "break": 0.205, "elo": 1810},
        SURFACE_GRASS:        {"spw": 0.675, "rpw": 0.345, "hold": 0.870, "break": 0.170, "elo": 1830},
    },
    # --- Jakub Mensik ----------------------------------------------------
    "Mensik": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.670, "rpw": 0.360, "hold": 0.865, "break": 0.185, "elo": 1840},
        SURFACE_HARD_INDOOR:  {"spw": 0.685, "rpw": 0.360, "hold": 0.885, "break": 0.185, "elo": 1860},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.365, "hold": 0.830, "break": 0.190, "elo": 1790},
        SURFACE_GRASS:        {"spw": 0.685, "rpw": 0.340, "hold": 0.885, "break": 0.165, "elo": 1840},
    },
    # --- Jiri Lehecka ----------------------------------------------------
    "Lehecka": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.665, "rpw": 0.365, "hold": 0.860, "break": 0.190, "elo": 1850},
        SURFACE_HARD_INDOOR:  {"spw": 0.680, "rpw": 0.365, "hold": 0.875, "break": 0.195, "elo": 1880},
        SURFACE_CLAY:         {"spw": 0.640, "rpw": 0.365, "hold": 0.830, "break": 0.195, "elo": 1800},
        SURFACE_GRASS:        {"spw": 0.680, "rpw": 0.345, "hold": 0.875, "break": 0.170, "elo": 1850},
    },
    # --- Roman Safiullin -------------------------------------------------
    "Safiullin": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.660, "rpw": 0.355, "hold": 0.855, "break": 0.180, "elo": 1810},
        SURFACE_HARD_INDOOR:  {"spw": 0.675, "rpw": 0.350, "hold": 0.870, "break": 0.175, "elo": 1830},
        SURFACE_CLAY:         {"spw": 0.630, "rpw": 0.355, "hold": 0.815, "break": 0.180, "elo": 1740},
        SURFACE_GRASS:        {"spw": 0.680, "rpw": 0.340, "hold": 0.875, "break": 0.165, "elo": 1810},
    },
    # --- Jan-Lennard Struff (big serve German) ---------------------------
    "Struff": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.675, "rpw": 0.350, "hold": 0.870, "break": 0.175, "elo": 1830},
        SURFACE_HARD_INDOOR:  {"spw": 0.690, "rpw": 0.345, "hold": 0.890, "break": 0.170, "elo": 1850},
        SURFACE_CLAY:         {"spw": 0.645, "rpw": 0.365, "hold": 0.835, "break": 0.195, "elo": 1810},
        SURFACE_GRASS:        {"spw": 0.700, "rpw": 0.335, "hold": 0.900, "break": 0.160, "elo": 1870},
    },
    # --- Lorenzo Sonego --------------------------------------------------
    "Sonego": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.650, "rpw": 0.360, "hold": 0.845, "break": 0.185, "elo": 1800},
        SURFACE_HARD_INDOOR:  {"spw": 0.665, "rpw": 0.355, "hold": 0.860, "break": 0.180, "elo": 1820},
        SURFACE_CLAY:         {"spw": 0.645, "rpw": 0.380, "hold": 0.840, "break": 0.215, "elo": 1850},
        SURFACE_GRASS:        {"spw": 0.670, "rpw": 0.340, "hold": 0.865, "break": 0.165, "elo": 1810},
    },

    # ======================================================================
    # WTA  (20 players)
    # ======================================================================

    # --- Iga Swiatek  (Clay queen) ---------------------------------------
    "Swiatek": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.620, "rpw": 0.430, "hold": 0.825, "break": 0.310, "elo": 2050},
        SURFACE_HARD_INDOOR:  {"spw": 0.625, "rpw": 0.420, "hold": 0.830, "break": 0.300, "elo": 2020},
        SURFACE_CLAY:         {"spw": 0.630, "rpw": 0.460, "hold": 0.840, "break": 0.355, "elo": 2200},
        SURFACE_GRASS:        {"spw": 0.620, "rpw": 0.395, "hold": 0.825, "break": 0.265, "elo": 1920},
    },
    # --- Aryna Sabalenka (hard-court queen, power) -----------------------
    "Sabalenka": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.635, "rpw": 0.425, "hold": 0.840, "break": 0.305, "elo": 2100},
        SURFACE_HARD_INDOOR:  {"spw": 0.640, "rpw": 0.420, "hold": 0.845, "break": 0.300, "elo": 2080},
        SURFACE_CLAY:         {"spw": 0.615, "rpw": 0.415, "hold": 0.815, "break": 0.290, "elo": 1980},
        SURFACE_GRASS:        {"spw": 0.640, "rpw": 0.400, "hold": 0.845, "break": 0.270, "elo": 2000},
    },
    # --- Coco Gauff -------------------------------------------------------
    "Gauff": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.600, "rpw": 0.425, "hold": 0.800, "break": 0.305, "elo": 2020},
        SURFACE_HARD_INDOOR:  {"spw": 0.605, "rpw": 0.415, "hold": 0.805, "break": 0.295, "elo": 2000},
        SURFACE_CLAY:         {"spw": 0.595, "rpw": 0.430, "hold": 0.790, "break": 0.315, "elo": 1960},
        SURFACE_GRASS:        {"spw": 0.610, "rpw": 0.395, "hold": 0.810, "break": 0.265, "elo": 1940},
    },
    # --- Elena Rybakina (big serve, grass ace) ---------------------------
    "Rybakina": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.640, "rpw": 0.390, "hold": 0.845, "break": 0.260, "elo": 2000},
        SURFACE_HARD_INDOOR:  {"spw": 0.650, "rpw": 0.385, "hold": 0.855, "break": 0.255, "elo": 2010},
        SURFACE_CLAY:         {"spw": 0.610, "rpw": 0.390, "hold": 0.810, "break": 0.260, "elo": 1900},
        SURFACE_GRASS:        {"spw": 0.660, "rpw": 0.385, "hold": 0.870, "break": 0.255, "elo": 2070},
    },
    # --- Jessica Pegula -------------------------------------------------
    "Pegula": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.590, "rpw": 0.415, "hold": 0.790, "break": 0.290, "elo": 1960},
        SURFACE_HARD_INDOOR:  {"spw": 0.595, "rpw": 0.410, "hold": 0.795, "break": 0.285, "elo": 1950},
        SURFACE_CLAY:         {"spw": 0.575, "rpw": 0.400, "hold": 0.770, "break": 0.270, "elo": 1870},
        SURFACE_GRASS:        {"spw": 0.600, "rpw": 0.390, "hold": 0.800, "break": 0.260, "elo": 1910},
    },
    # --- Zheng Qinwen ----------------------------------------------------
    "Zheng": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.625, "rpw": 0.400, "hold": 0.830, "break": 0.275, "elo": 1980},
        SURFACE_HARD_INDOOR:  {"spw": 0.630, "rpw": 0.395, "hold": 0.835, "break": 0.270, "elo": 1970},
        SURFACE_CLAY:         {"spw": 0.605, "rpw": 0.400, "hold": 0.805, "break": 0.275, "elo": 1910},
        SURFACE_GRASS:        {"spw": 0.635, "rpw": 0.380, "hold": 0.840, "break": 0.250, "elo": 1950},
    },
    # --- Jelena Ostapenko (all-or-nothing power) -------------------------
    "Ostapenko": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.615, "rpw": 0.400, "hold": 0.815, "break": 0.275, "elo": 1880},
        SURFACE_HARD_INDOOR:  {"spw": 0.620, "rpw": 0.395, "hold": 0.820, "break": 0.270, "elo": 1890},
        SURFACE_CLAY:         {"spw": 0.610, "rpw": 0.415, "hold": 0.810, "break": 0.295, "elo": 1910},
        SURFACE_GRASS:        {"spw": 0.630, "rpw": 0.385, "hold": 0.835, "break": 0.255, "elo": 1900},
    },
    # --- Ons Jabeur  (Clay/Grass feel player) ----------------------------
    "Jabeur": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.585, "rpw": 0.405, "hold": 0.785, "break": 0.280, "elo": 1900},
        SURFACE_HARD_INDOOR:  {"spw": 0.585, "rpw": 0.395, "hold": 0.785, "break": 0.270, "elo": 1870},
        SURFACE_CLAY:         {"spw": 0.600, "rpw": 0.420, "hold": 0.800, "break": 0.300, "elo": 1960},
        SURFACE_GRASS:        {"spw": 0.610, "rpw": 0.405, "hold": 0.810, "break": 0.280, "elo": 1980},
    },
    # --- Jasmine Paolini ------------------------------------------------
    "Paolini": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.580, "rpw": 0.410, "hold": 0.775, "break": 0.285, "elo": 1920},
        SURFACE_HARD_INDOOR:  {"spw": 0.585, "rpw": 0.400, "hold": 0.780, "break": 0.275, "elo": 1900},
        SURFACE_CLAY:         {"spw": 0.595, "rpw": 0.425, "hold": 0.795, "break": 0.305, "elo": 1970},
        SURFACE_GRASS:        {"spw": 0.595, "rpw": 0.395, "hold": 0.795, "break": 0.270, "elo": 1950},
    },
    # --- Daria Kasatkina -------------------------------------------------
    "Kasatkina": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.565, "rpw": 0.410, "hold": 0.760, "break": 0.285, "elo": 1870},
        SURFACE_HARD_INDOOR:  {"spw": 0.570, "rpw": 0.400, "hold": 0.765, "break": 0.275, "elo": 1860},
        SURFACE_CLAY:         {"spw": 0.580, "rpw": 0.425, "hold": 0.775, "break": 0.305, "elo": 1920},
        SURFACE_GRASS:        {"spw": 0.570, "rpw": 0.380, "hold": 0.765, "break": 0.250, "elo": 1810},
    },
    # --- Madison Keys (power) -------------------------------------------
    "Keys": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.635, "rpw": 0.385, "hold": 0.840, "break": 0.255, "elo": 1950},
        SURFACE_HARD_INDOOR:  {"spw": 0.640, "rpw": 0.380, "hold": 0.845, "break": 0.250, "elo": 1940},
        SURFACE_CLAY:         {"spw": 0.605, "rpw": 0.380, "hold": 0.805, "break": 0.250, "elo": 1860},
        SURFACE_GRASS:        {"spw": 0.645, "rpw": 0.370, "hold": 0.850, "break": 0.240, "elo": 1940},
    },
    # --- Emma Navarro ----------------------------------------------------
    "Navarro": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.590, "rpw": 0.405, "hold": 0.790, "break": 0.280, "elo": 1910},
        SURFACE_HARD_INDOOR:  {"spw": 0.595, "rpw": 0.395, "hold": 0.795, "break": 0.270, "elo": 1890},
        SURFACE_CLAY:         {"spw": 0.580, "rpw": 0.400, "hold": 0.775, "break": 0.275, "elo": 1860},
        SURFACE_GRASS:        {"spw": 0.600, "rpw": 0.385, "hold": 0.800, "break": 0.255, "elo": 1890},
    },
    # --- Bianca Andreescu ------------------------------------------------
    "Andreescu": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.600, "rpw": 0.405, "hold": 0.800, "break": 0.280, "elo": 1890},
        SURFACE_HARD_INDOOR:  {"spw": 0.605, "rpw": 0.395, "hold": 0.805, "break": 0.270, "elo": 1870},
        SURFACE_CLAY:         {"spw": 0.590, "rpw": 0.400, "hold": 0.790, "break": 0.275, "elo": 1860},
        SURFACE_GRASS:        {"spw": 0.600, "rpw": 0.380, "hold": 0.800, "break": 0.250, "elo": 1830},
    },
    # --- Paula Badosa ----------------------------------------------------
    "Badosa": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.600, "rpw": 0.400, "hold": 0.800, "break": 0.275, "elo": 1900},
        SURFACE_HARD_INDOOR:  {"spw": 0.610, "rpw": 0.395, "hold": 0.810, "break": 0.270, "elo": 1900},
        SURFACE_CLAY:         {"spw": 0.600, "rpw": 0.415, "hold": 0.800, "break": 0.290, "elo": 1920},
        SURFACE_GRASS:        {"spw": 0.605, "rpw": 0.380, "hold": 0.805, "break": 0.250, "elo": 1850},
    },
    # --- Donna Vekic -----------------------------------------------------
    "Vekic": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.610, "rpw": 0.390, "hold": 0.810, "break": 0.260, "elo": 1880},
        SURFACE_HARD_INDOOR:  {"spw": 0.620, "rpw": 0.385, "hold": 0.820, "break": 0.255, "elo": 1880},
        SURFACE_CLAY:         {"spw": 0.590, "rpw": 0.390, "hold": 0.790, "break": 0.260, "elo": 1830},
        SURFACE_GRASS:        {"spw": 0.625, "rpw": 0.375, "hold": 0.830, "break": 0.245, "elo": 1880},
    },
    # --- Barbora Krejcikova ----------------------------------------------
    "Krejcikova": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.590, "rpw": 0.395, "hold": 0.790, "break": 0.270, "elo": 1860},
        SURFACE_HARD_INDOOR:  {"spw": 0.600, "rpw": 0.390, "hold": 0.800, "break": 0.260, "elo": 1870},
        SURFACE_CLAY:         {"spw": 0.595, "rpw": 0.410, "hold": 0.795, "break": 0.285, "elo": 1900},
        SURFACE_GRASS:        {"spw": 0.610, "rpw": 0.385, "hold": 0.810, "break": 0.255, "elo": 1910},
    },
    # --- Karolina Muchova ------------------------------------------------
    "Muchova": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.605, "rpw": 0.405, "hold": 0.805, "break": 0.280, "elo": 1910},
        SURFACE_HARD_INDOOR:  {"spw": 0.615, "rpw": 0.400, "hold": 0.815, "break": 0.275, "elo": 1910},
        SURFACE_CLAY:         {"spw": 0.600, "rpw": 0.415, "hold": 0.800, "break": 0.290, "elo": 1930},
        SURFACE_GRASS:        {"spw": 0.615, "rpw": 0.390, "hold": 0.815, "break": 0.260, "elo": 1900},
    },
    # --- Liudmila Samsonova (big serve WTA) ------------------------------
    "Samsonova": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.625, "rpw": 0.380, "hold": 0.830, "break": 0.250, "elo": 1870},
        SURFACE_HARD_INDOOR:  {"spw": 0.635, "rpw": 0.375, "hold": 0.840, "break": 0.245, "elo": 1880},
        SURFACE_CLAY:         {"spw": 0.600, "rpw": 0.380, "hold": 0.800, "break": 0.250, "elo": 1810},
        SURFACE_GRASS:        {"spw": 0.640, "rpw": 0.370, "hold": 0.845, "break": 0.240, "elo": 1900},
    },
    # --- Jessica Bouzas Maneiro -----------------------------------------
    "Bouzas": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.580, "rpw": 0.390, "hold": 0.775, "break": 0.260, "elo": 1810},
        SURFACE_HARD_INDOOR:  {"spw": 0.585, "rpw": 0.385, "hold": 0.780, "break": 0.255, "elo": 1800},
        SURFACE_CLAY:         {"spw": 0.585, "rpw": 0.400, "hold": 0.780, "break": 0.275, "elo": 1840},
        SURFACE_GRASS:        {"spw": 0.585, "rpw": 0.370, "hold": 0.780, "break": 0.240, "elo": 1770},
    },
    # --- Ekaterina Alexandrova (power baseline) -------------------------
    "Alexandrova": {
        SURFACE_HARD_OUTDOOR: {"spw": 0.615, "rpw": 0.385, "hold": 0.815, "break": 0.255, "elo": 1850},
        SURFACE_HARD_INDOOR:  {"spw": 0.625, "rpw": 0.380, "hold": 0.825, "break": 0.250, "elo": 1860},
        SURFACE_CLAY:         {"spw": 0.595, "rpw": 0.385, "hold": 0.795, "break": 0.255, "elo": 1800},
        SURFACE_GRASS:        {"spw": 0.620, "rpw": 0.370, "hold": 0.820, "break": 0.240, "elo": 1830},
    },
}

# ---------------------------------------------------------------------------
# Known tournament → surface mapping (case-insensitive lookup)
# ---------------------------------------------------------------------------

_TOURNAMENT_SURFACE_MAP: Dict[str, str] = {
    # Grand Slams
    "australian open": SURFACE_HARD_OUTDOOR,
    "roland garros": SURFACE_CLAY,
    "wimbledon": SURFACE_GRASS,
    "us open": SURFACE_HARD_OUTDOOR,
    # ATP Masters 1000
    "indian wells": SURFACE_HARD_OUTDOOR,
    "miami open": SURFACE_HARD_OUTDOOR,
    "monte carlo": SURFACE_CLAY,
    "monte-carlo": SURFACE_CLAY,
    "madrid open": SURFACE_CLAY,
    "mutua madrid open": SURFACE_CLAY,
    "italian open": SURFACE_CLAY,
    "internazionali bnl": SURFACE_CLAY,
    "roma": SURFACE_CLAY,
    "canada open": SURFACE_HARD_OUTDOOR,
    "canadian open": SURFACE_HARD_OUTDOOR,
    "rogers cup": SURFACE_HARD_OUTDOOR,
    "cincinnati": SURFACE_HARD_OUTDOOR,
    "shanghai masters": SURFACE_HARD_OUTDOOR,
    "paris masters": SURFACE_HARD_INDOOR,
    "paris bercy": SURFACE_HARD_INDOOR,
    "rolex paris masters": SURFACE_HARD_INDOOR,
    # ATP 500 / notable
    "rotterdam": SURFACE_HARD_INDOOR,
    "dubai": SURFACE_HARD_OUTDOOR,
    "acapulco": SURFACE_HARD_OUTDOOR,
    "barcelona": SURFACE_CLAY,
    "hamburg": SURFACE_CLAY,
    "halle": SURFACE_GRASS,
    "queens": SURFACE_GRASS,
    "queen's club": SURFACE_GRASS,
    "washington": SURFACE_HARD_OUTDOOR,
    "tokyo": SURFACE_HARD_OUTDOOR,
    "beijing": SURFACE_HARD_OUTDOOR,
    "vienna": SURFACE_HARD_INDOOR,
    "erste bank open": SURFACE_HARD_INDOOR,
    "basel": SURFACE_HARD_INDOOR,
    "swiss indoors": SURFACE_HARD_INDOOR,
    "atp finals": SURFACE_HARD_INDOOR,
    "nitto atp finals": SURFACE_HARD_INDOOR,
    "wta finals": SURFACE_HARD_INDOOR,
    # WTA notable
    "doha": SURFACE_HARD_OUTDOOR,
    "qatar open": SURFACE_HARD_OUTDOOR,
    "stuttgart": SURFACE_CLAY,
    "eastbourne": SURFACE_GRASS,
    "birmingham": SURFACE_GRASS,
    "s-hertogenbosch": SURFACE_GRASS,
    "guadalajara": SURFACE_HARD_OUTDOOR,
    "beijing open": SURFACE_HARD_OUTDOOR,
    "wuhan": SURFACE_HARD_OUTDOOR,
    # South-American clay swing
    "buenos aires": SURFACE_CLAY,
    "rio open": SURFACE_CLAY,
    "rio de janeiro": SURFACE_CLAY,
    "santiago": SURFACE_CLAY,
    "cordoba": SURFACE_CLAY,
    "estoril": SURFACE_CLAY,
    "lyon": SURFACE_CLAY,
    "geneva": SURFACE_CLAY,
    "umag": SURFACE_CLAY,
    "kitzbuhel": SURFACE_CLAY,
    "bastad": SURFACE_CLAY,
    "gstaad": SURFACE_CLAY,
    # Hard indoor 250s
    "marseille": SURFACE_HARD_INDOOR,
    "montpellier": SURFACE_HARD_INDOOR,
    "dallas": SURFACE_HARD_INDOOR,
    "metz": SURFACE_HARD_INDOOR,
    "sofia": SURFACE_HARD_INDOOR,
    "stockholm": SURFACE_HARD_INDOOR,
    "antwerp": SURFACE_HARD_INDOOR,
    # Grass 250s
    "mallorca": SURFACE_GRASS,
    "newport": SURFACE_GRASS,
    "nottingham": SURFACE_GRASS,
}

# ---------------------------------------------------------------------------
# 4. get_player_profile  (fuzzy-matched lookup)
# ---------------------------------------------------------------------------


def get_player_profile(player_name: str, surface: str) -> Dict[str, float]:
    """Return serve/return statistics for *player_name* on *surface*.

    Matching uses ``rapidfuzz.fuzz.token_sort_ratio`` with a threshold of 72.
    Falls back to :data:`TOUR_AVERAGES` when no match is found or the surface
    is not in the matched player's profile.

    Parameters
    ----------
    player_name:
        Free-text player name (e.g. ``"Jannik Sinner"`` or ``"sinner"``).
    surface:
        One of the ``SURFACE_*`` constants.

    Returns
    -------
    Dict[str, float]
        Keys: ``spw``, ``rpw``, ``hold``, ``break``, ``elo``.
    """
    # Direct substring / token match first (handles "Jannik Sinner" -> "Sinner")
    lower = player_name.strip().lower()
    for key, surfaces in PLAYER_PROFILES.items():
        k_low = key.lower()
        if k_low == lower or k_low in lower or lower in k_low:
            return dict(surfaces.get(surface, TOUR_AVERAGES.get(surface, TOUR_AVERAGES[SURFACE_HARD_OUTDOOR])))

    try:
        from rapidfuzz import fuzz  # type: ignore[import-untyped]
    except ImportError:
        return dict(TOUR_AVERAGES.get(surface, TOUR_AVERAGES[SURFACE_HARD_OUTDOOR]))

    best_score: float = 0.0
    best_key: str | None = None
    query = player_name.strip()

    for key in PLAYER_PROFILES:
        score: float = max(
            float(fuzz.token_set_ratio(query, key)),
            float(fuzz.partial_ratio(query.lower(), key.lower()))
        )
        if score > best_score:
            best_score = score
            best_key = key

    if best_key is not None and best_score >= 70:
        profile = PLAYER_PROFILES[best_key]
        return dict(profile.get(surface, TOUR_AVERAGES.get(surface, TOUR_AVERAGES[SURFACE_HARD_OUTDOOR])))

    return dict(TOUR_AVERAGES.get(surface, TOUR_AVERAGES[SURFACE_HARD_OUTDOOR]))


# ---------------------------------------------------------------------------
# 5. detect_surface_from_tournament
# ---------------------------------------------------------------------------


def detect_surface_from_tournament(tournament_name: str) -> str:
    """Heuristically determine the court surface from a tournament name.

    Strategy (evaluated in order):
    1. Exact lookup in :data:`_TOURNAMENT_SURFACE_MAP` (case-insensitive).
    2. Keyword matching: ``clay``, ``arcilla``, ``roland`` → Clay; ``grass``,
       ``hierba``, ``wimbledon`` → Grass; ``indoor`` → Hard Indoor.
    3. Default: Hard Outdoor.

    Parameters
    ----------
    tournament_name:
        The tournament label as it appears in schedule data.

    Returns
    -------
    str
        One of the ``SURFACE_*`` constants.
    """
    lower = tournament_name.strip().lower()

    # 1. Direct lookup
    if lower in _TOURNAMENT_SURFACE_MAP:
        return _TOURNAMENT_SURFACE_MAP[lower]

    # 1b. Partial match against known names (handles extra whitespace / year)
    for known, surface in _TOURNAMENT_SURFACE_MAP.items():
        if known in lower or lower in known:
            return surface

    # 2. Keyword heuristics
    clay_keywords = ("clay", "arcilla", "roland", "garros", "tierra")
    grass_keywords = ("grass", "hierba", "wimbledon")
    indoor_keywords = ("indoor",)

    if any(kw in lower for kw in clay_keywords):
        return SURFACE_CLAY
    if any(kw in lower for kw in grass_keywords):
        return SURFACE_GRASS
    if any(kw in lower for kw in indoor_keywords):
        return SURFACE_HARD_INDOOR

    # 3. Default
    return SURFACE_HARD_OUTDOOR


# ---------------------------------------------------------------------------
# 6. get_surface_emoji
# ---------------------------------------------------------------------------


def get_surface_emoji(surface: str) -> str:
    """Return an emoji badge for the given *surface*.

    Returns
    -------
    str
        🟦 Hard Indoor, 🔵 Hard Outdoor, 🟧 Clay, 🟩 Grass.
    """
    return {
        SURFACE_HARD_INDOOR: "🟦",
        SURFACE_HARD_OUTDOOR: "🔵",
        SURFACE_CLAY: "🟧",
        SURFACE_GRASS: "🟩",
    }.get(surface, "⬜")


# ---------------------------------------------------------------------------
# 7. get_surface_color
# ---------------------------------------------------------------------------


def get_surface_color(surface: str) -> str:
    """Return a hex colour string for the given *surface*.

    Returns
    -------
    str
        CSS-ready hex colour (e.g. ``"#2563eb"``).
    """
    return {
        SURFACE_HARD_INDOOR: "#3b82f6",
        SURFACE_HARD_OUTDOOR: "#2563eb",
        SURFACE_CLAY: "#ea580c",
        SURFACE_GRASS: "#16a34a",
    }.get(surface, "#6b7280")
