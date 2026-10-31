"""
database/db_manager.py
~~~~~~~~~~~~~~~~~~~~~~
SQLite CRUD manager for QuantCourt AI – tennis betting tracker.

Provides helpers to create tables, insert/update/delete bets,
manage bankroll configuration, and compute performance statistics.

All public functions open their own connection, execute, and close,
so callers never need to manage connections manually.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Database path – lives in the project root as ``quantcourt.db``
# ---------------------------------------------------------------------------
DB_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "quantcourt.db",
)

# ---------------------------------------------------------------------------
# Valid bet statuses
# ---------------------------------------------------------------------------
VALID_STATUSES: tuple[str, ...] = (
    "Pendiente",
    "Ganada",
    "Perdida",
    "Push",
    "Retirada",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with *Row* factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a ``sqlite3.Row`` to a plain dictionary."""
    return dict(row)


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the required tables if they do not already exist.

    Tables
    ------
    * ``tennis_bets`` – every individual bet record.
    * ``tennis_bankroll_config`` – single-row bankroll configuration.
    """
    conn = _get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tennis_bets (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   TEXT    DEFAULT (datetime('now','localtime')),
                sport        TEXT    DEFAULT 'Tenis',
                tournament   TEXT,
                surface      TEXT,
                match_name   TEXT,
                market       TEXT,
                selection    TEXT,
                bookmaker    TEXT    DEFAULT 'BetPlay',
                odds         REAL,
                fair_odds    REAL,
                ev_percent   REAL,
                stake        REAL,
                status       TEXT    DEFAULT 'Pendiente'
                             CHECK (status IN ('Pendiente','Ganada','Perdida','Push','Retirada')),
                payout       REAL    DEFAULT 0.0,
                profit       REAL    DEFAULT 0.0,
                notes        TEXT    DEFAULT '',
                ai_analysis  TEXT    DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS tennis_bankroll_config (
                id                INTEGER PRIMARY KEY,
                initial_bankroll  REAL    DEFAULT 100000.0,
                currency          TEXT    DEFAULT 'COP'
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Bet CRUD
# ---------------------------------------------------------------------------

def add_bet(
    tournament: str,
    surface: str,
    match_name: str,
    market: str,
    selection: str,
    odds: float,
    fair_odds: float,
    ev_percent: float,
    stake: float,
    bookmaker: str = "BetPlay",
    notes: str = "",
    ai_analysis: str = "",
) -> int:
    """Insert a new bet and return the auto-generated row *id*.

    Parameters
    ----------
    tournament : str
        Name of the tournament (e.g. ``"ATP Miami Open 2025"``).
    surface : str
        Court surface (``'Dura Indoor'``, ``'Arcilla'``, ``'Hierba'``,
        ``'Dura Outdoor'``).
    match_name : str
        Players involved (e.g. ``"Djokovic vs Sinner"``).
    market : str
        Bet market (e.g. ``"ML"`` for money-line, ``"Handicap"``).
    selection : str
        Picked outcome (e.g. ``"Djokovic"``).
    odds : float
        Decimal odds offered by the bookmaker.
    fair_odds : float
        Model-estimated fair decimal odds.
    ev_percent : float
        Expected Value percentage.
    stake : float
        Amount wagered in the bankroll currency.
    bookmaker : str, optional
        Sportsbook used (default ``'BetPlay'``).
    notes : str, optional
        Free-form user notes.
    ai_analysis : str, optional
        AI-generated analysis attached to the bet.

    Returns
    -------
    int
        The ``id`` of the newly inserted row.
    """
    sql = """
        INSERT INTO tennis_bets
            (tournament, surface, match_name, market, selection,
             bookmaker, odds, fair_odds, ev_percent, stake,
             notes, ai_analysis)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn = _get_connection()
    try:
        cur = conn.execute(
            sql,
            (
                tournament,
                surface,
                match_name,
                market,
                selection,
                bookmaker,
                odds,
                fair_odds,
                ev_percent,
                stake,
                notes,
                ai_analysis,
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def get_all_bets() -> List[Dict[str, Any]]:
    """Return every bet as a list of dictionaries, newest first."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM tennis_bets ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def update_bet_status(
    bet_id: int,
    new_status: str,
    payout: Optional[float] = None,
) -> None:
    """Update the status of a bet and recalculate its profit.

    Parameters
    ----------
    bet_id : int
        Primary key of the bet to update.
    new_status : str
        Must be one of ``VALID_STATUSES``.
    payout : float or None
        Explicit payout amount.  When *None* the payout is derived
        automatically:

        * **Ganada** → ``stake × odds``
        * **Perdida** → ``0.0``
        * **Push** → ``stake`` (money returned)
        * **Retirada** → ``stake`` (money returned)
        * **Pendiente** → ``0.0``

    Raises
    ------
    ValueError
        If *new_status* is not in ``VALID_STATUSES``.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Estado inválido '{new_status}'. "
            f"Debe ser uno de {VALID_STATUSES}"
        )

    conn = _get_connection()
    try:
        # Fetch the existing bet to know stake & odds
        row = conn.execute(
            "SELECT stake, odds FROM tennis_bets WHERE id = ?", (bet_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No se encontró la apuesta con id={bet_id}")

        stake: float = row["stake"]
        odds: float = row["odds"]

        # Derive payout when not provided explicitly
        if payout is None:
            if new_status == "Ganada":
                payout = round(stake * odds, 2)
            elif new_status in ("Push", "Retirada"):
                payout = stake
            else:
                payout = 0.0

        profit = round(payout - stake, 2)

        conn.execute(
            """
            UPDATE tennis_bets
               SET status = ?,
                   payout = ?,
                   profit = ?
             WHERE id = ?
            """,
            (new_status, payout, profit, bet_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_bet(bet_id: int) -> None:
    """Delete a bet by its primary key.

    Parameters
    ----------
    bet_id : int
        Row ``id`` to remove.
    """
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM tennis_bets WHERE id = ?", (bet_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Bankroll configuration
# ---------------------------------------------------------------------------

def get_bankroll_config() -> Dict[str, Any]:
    """Return the bankroll configuration as a dictionary.

    If no configuration row exists yet, a default one is inserted
    (``initial_bankroll=100000.0``, ``currency='COP'``).

    Returns
    -------
    dict
        Keys: ``initial_bankroll`` (float), ``currency`` (str).
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT initial_bankroll, currency FROM tennis_bankroll_config WHERE id = 1"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO tennis_bankroll_config (id, initial_bankroll, currency) "
                "VALUES (1, 100000.0, 'COP')"
            )
            conn.commit()
            return {"initial_bankroll": 100000.0, "currency": "COP"}
        return _row_to_dict(row)
    finally:
        conn.close()


def set_initial_bankroll(amount: float, currency: str = "COP") -> None:
    """Insert or update the bankroll configuration (upsert).

    Parameters
    ----------
    amount : float
        New initial bankroll value.
    currency : str, optional
        Currency code (default ``'COP'``).
    """
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO tennis_bankroll_config (id, initial_bankroll, currency)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE
               SET initial_bankroll = excluded.initial_bankroll,
                   currency         = excluded.currency
            """,
            (amount, currency),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def calculate_current_bankroll() -> float:
    """Return the current bankroll (initial + cumulative profit).

    Returns
    -------
    float
        Current bankroll value rounded to two decimals.
    """
    config = get_bankroll_config()
    initial: float = config["initial_bankroll"]

    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(profit), 0.0) AS total_profit FROM tennis_bets"
        ).fetchone()
        total_profit: float = row["total_profit"] if row else 0.0
        return round(initial + total_profit, 2)
    finally:
        conn.close()


def get_performance_stats() -> Dict[str, Any]:
    """Compute aggregate performance statistics across all bets.

    Returns
    -------
    dict
        Keys:

        * ``total_bets`` – count of all bets
        * ``won`` – count with status ``'Ganada'``
        * ``lost`` – count with status ``'Perdida'``
        * ``pushed`` – count with status ``'Push'``
        * ``pending`` – count with status ``'Pendiente'``
        * ``win_rate`` – won / resolved bets (%) or 0.0
        * ``total_staked`` – sum of stakes
        * ``total_profit`` – sum of profits
        * ``roi_pct`` – total_profit / total_staked × 100 (or 0.0)
        * ``yield_pct`` – alias for roi_pct (profit per unit staked)
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                              AS total_bets,
                SUM(CASE WHEN status = 'Ganada'    THEN 1 ELSE 0 END) AS won,
                SUM(CASE WHEN status = 'Perdida'   THEN 1 ELSE 0 END) AS lost,
                SUM(CASE WHEN status = 'Push'      THEN 1 ELSE 0 END) AS pushed,
                SUM(CASE WHEN status = 'Pendiente' THEN 1 ELSE 0 END) AS pending,
                COALESCE(SUM(stake),  0.0)                            AS total_staked,
                COALESCE(SUM(profit), 0.0)                            AS total_profit
            FROM tennis_bets
            """
        ).fetchone()

        assert row is not None  # always returns a row (counts may be 0)

        total_bets: int = int(row["total_bets"] or 0)
        won: int = int(row["won"] or 0)
        lost: int = int(row["lost"] or 0)
        pushed: int = int(row["pushed"] or 0)
        pending: int = int(row["pending"] or 0)
        total_staked: float = float(row["total_staked"] or 0.0)
        total_profit: float = float(row["total_profit"] or 0.0)

        resolved = won + lost + pushed
        win_rate = round((won / resolved) * 100, 2) if resolved > 0 else 0.0
        roi_pct = round((total_profit / total_staked) * 100, 2) if total_staked > 0 else 0.0

        return {
            "total_bets": total_bets,
            "won": won,
            "lost": lost,
            "pushed": pushed,
            "pending": pending,
            "win_rate": win_rate,
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": roi_pct,
            "yield_pct": roi_pct,
        }
    finally:
        conn.close()


def get_bankroll_evolution() -> List[Dict[str, Any]]:
    """Build a chronological list of bankroll snapshots for charting.

    Each entry contains:

    * ``date`` – the ``created_at`` date (``YYYY-MM-DD``).
    * ``cumulative_profit`` – running sum of profits up to (and including)
      that bet.
    * ``bankroll`` – ``initial_bankroll + cumulative_profit``.

    Only resolved bets (status ≠ ``'Pendiente'``) are included.

    Returns
    -------
    list[dict]
        Ordered by ``created_at ASC``.
    """
    config = get_bankroll_config()
    initial: float = config["initial_bankroll"]

    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                DATE(created_at) AS date,
                profit
            FROM tennis_bets
            WHERE status != 'Pendiente'
            ORDER BY created_at ASC
            """
        ).fetchall()

        evolution: List[Dict[str, Any]] = []
        cumulative: float = 0.0
        for r in rows:
            cumulative += r["profit"]
            evolution.append(
                {
                    "date": r["date"],
                    "cumulative_profit": round(cumulative, 2),
                    "bankroll": round(initial + cumulative, 2),
                }
            )
        return evolution
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Auto-initialise on import
# ---------------------------------------------------------------------------
init_db()
