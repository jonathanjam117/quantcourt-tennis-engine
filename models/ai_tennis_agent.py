"""
models/ai_tennis_agent.py
─────────────────────────
Gemini AI tactical agent for tennis match evaluation.

Provides AI-driven tactical adjustments (serve multipliers, fatigue estimates,
surface advantages, H2H bias) consumed by the Kelly / EV pipeline.  Every
public entry point is wrapped in fail-safe logic so the caller **never**
receives ``None`` or an unexpected exception — only a valid
``TennisTacticalAdjustment`` instance.

QuantCourt AI — © 2024-2026
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic model — tactical adjustment payload
# ---------------------------------------------------------------------------


class TennisTacticalAdjustment(BaseModel):
    """AI-generated tactical adjustment for a single tennis match.

    Every field carries a sensible default so that a bare
    ``TennisTacticalAdjustment()`` is always a valid *neutral* adjustment.

    The ``__getitem__`` / ``get`` dunder methods allow dict-style access,
    providing full **blindaje** against ``TypeError`` when downstream code
    does ``obj["key"]`` or ``obj.get("key")``.
    """

    factor_saque_a: float = Field(
        default=1.0,
        description="Multiplier for player A's serve strength.",
    )
    factor_saque_b: float = Field(
        default=1.0,
        description="Multiplier for player B's serve strength.",
    )
    confianza_ia: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="AI confidence level (0.0 – 1.0).",
    )
    nivel_riesgo: str = Field(
        default="Bajo",
        description='Risk level: "Bajo", "Medio", or "Alto".',
    )
    justificacion_tactica: str = Field(
        default="Análisis estadístico base sin alteraciones tácticas.",
        description="Free-text tactical justification produced by the AI.",
    )
    fatiga_a: str = Field(
        default="Normal",
        description='Player A fatigue: "Fresco", "Normal", or "Desgastado".',
    )
    fatiga_b: str = Field(
        default="Normal",
        description='Player B fatigue: "Fresco", "Normal", or "Desgastado".',
    )
    ventaja_superficie: str = Field(
        default="Neutral",
        description='Surface advantage: "Jugador A", "Jugador B", or "Neutral".',
    )
    h2h_favorece: str = Field(
        default="Neutral",
        description='H2H bias: "Jugador A", "Jugador B", or "Neutral".',
    )

    # ── Blindaje: dict-style access ────────────────────────────────────────

    def __getitem__(self, item: str) -> Any:
        """Allow ``obj["key"]`` access without raising ``TypeError``."""
        return getattr(self, item, None)

    def get(self, item: str, default: Any = None) -> Any:
        """Allow ``obj.get("key", default)`` access like a regular dict."""
        return getattr(self, item, default)


# ---------------------------------------------------------------------------
# Module-level neutral default (singleton)
# ---------------------------------------------------------------------------

DEFAULT_TACTICAL_ADJUSTMENT: TennisTacticalAdjustment = TennisTacticalAdjustment()
"""Neutral adjustment returned whenever the AI call is skipped or fails."""


# ---------------------------------------------------------------------------
# Universal safe accessor
# ---------------------------------------------------------------------------


def safe_ai_get(obj: Any, key: str, default: Any = "") -> Any:
    """Extract *key* from *obj* regardless of its type.

    Works transparently with ``None``, plain ``dict``, and any Pydantic /
    dataclass object that exposes attributes.

    Parameters
    ----------
    obj:
        The source object (may be ``None``).
    key:
        Attribute / dict key to look up.
    default:
        Value returned when the key is missing or *obj* is ``None``.

    Returns
    -------
    Any
        The resolved value, or *default*.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Available Gemini model identifiers
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
]


# ---------------------------------------------------------------------------
# Core AI evaluation function
# ---------------------------------------------------------------------------


def evaluar_contexto_partido_tenis(
    torneo: str,
    superficie: str,
    jugador_a: str,
    jugador_b: str,
    cuotas_actuales: Optional[Dict[str, Any]] = None,
    contexto_usuario: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    is_demo: bool = False,
) -> TennisTacticalAdjustment:
    """Ask Gemini to evaluate the tactical context of a tennis match.

    The function builds a structured prompt that asks the model to assess:

    * **Physical fatigue** — matches played in the last 48 h.
    * **Surface adaptation** — clay / grass / hard-court affinity.
    * **Head-to-head dominance** — historical H2H record bias.
    * **Altitude / climate effects** — ball speed, bounce height.
    * **Travel schedule** — jet-lag and recovery concerns.

    Parameters
    ----------
    torneo:
        Tournament name (e.g. ``"Roland Garros 2026"``).
    superficie:
        Court surface (``"Arcilla"``, ``"Hierba"``, ``"Dura"``).
    jugador_a:
        Full name of player A.
    jugador_b:
        Full name of player B.
    cuotas_actuales:
        Optional dict with current odds, e.g.
        ``{"cuota_a": 1.65, "cuota_b": 2.30}``.
    contexto_usuario:
        Optional free-text context supplied by the analyst.
    api_key:
        Google Gemini API key.  If absent the demo default is returned.
    model_name:
        Gemini model identifier (default ``"gemini-2.5-flash"``).
    is_demo:
        When ``True`` the AI call is skipped and a neutral default returned.

    Returns
    -------
    TennisTacticalAdjustment
        A validated tactical adjustment.  **Never** ``None``, **never**
        raises an exception.
    """
    # ── Fast-path: demo mode or missing credentials ────────────────────────
    try:
        if is_demo or not api_key:
            return DEFAULT_TACTICAL_ADJUSTMENT

        # ── Build the evaluation prompt ────────────────────────────────────
        cuotas_str = ""
        if cuotas_actuales:
            cuotas_str = (
                f"\nCuotas actuales del mercado: {json.dumps(cuotas_actuales, ensure_ascii=False)}"
            )

        contexto_str = ""
        if contexto_usuario:
            contexto_str = f"\nContexto adicional del analista: {contexto_usuario}"

        prompt = (
            f"Eres un analista táctico experto de tenis profesional.\n"
            f"Evalúa el siguiente partido y devuelve ÚNICAMENTE un JSON válido "
            f"(sin bloques de código, sin comentarios, sin texto adicional).\n\n"
            f"Torneo: {torneo}\n"
            f"Superficie: {superficie}\n"
            f"Jugador A: {jugador_a}\n"
            f"Jugador B: {jugador_b}"
            f"{cuotas_str}"
            f"{contexto_str}\n\n"
            f"Evalúa los siguientes factores:\n"
            f"1. Fatiga física — partidos jugados en las últimas 48 horas.\n"
            f"2. Adaptación a la superficie — afinidad de cada jugador.\n"
            f"3. Dominancia head-to-head — historial H2H entre ambos.\n"
            f"4. Efectos de altitud/clima — velocidad de bola, altura de bote.\n"
            f"5. Agenda de viaje — jet-lag y recuperación.\n\n"
            f"Responde SOLO con un objeto JSON con exactamente estos campos:\n"
            f'{{\n'
            f'  "factor_saque_a": <float, multiplicador de saque de A, base 1.0>,\n'
            f'  "factor_saque_b": <float, multiplicador de saque de B, base 1.0>,\n'
            f'  "confianza_ia": <float entre 0.0 y 1.0>,\n'
            f'  "nivel_riesgo": "<Bajo | Medio | Alto>",\n'
            f'  "justificacion_tactica": "<texto explicativo>",\n'
            f'  "fatiga_a": "<Fresco | Normal | Desgastado>",\n'
            f'  "fatiga_b": "<Fresco | Normal | Desgastado>",\n'
            f'  "ventaja_superficie": "<Jugador A | Jugador B | Neutral>",\n'
            f'  "h2h_favorece": "<Jugador A | Jugador B | Neutral>"\n'
            f'}}'
        )

        # ── Call Gemini via the new google.genai SDK ───────────────────────
        import google.genai  # type: ignore[import-untyped]

        client = google.genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        # ── Parse the response text ───────────────────────────────────────
        raw_text: str = response.text or ""

        # Strip optional markdown code fences (```json ... ```)
        cleaned = raw_text.strip()
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            cleaned,
            re.DOTALL,
        )
        if fence_match:
            cleaned = fence_match.group(1).strip()

        parsed_data: dict[str, Any] = json.loads(cleaned)

        # ── Validate with Pydantic ────────────────────────────────────────
        adjustment = TennisTacticalAdjustment(**parsed_data)
        return adjustment

    except Exception:  # noqa: BLE001 — intentional catch-all blindaje
        return DEFAULT_TACTICAL_ADJUSTMENT
