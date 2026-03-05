# ============================================================
#  strategy.py  –  SMC Strategy Engine
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import (
    HTF, LTF, HTF_LIMIT, LTF_LIMIT,
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    MIN_RR, SL_BUFFER_PCT, MIN_SCORE,
)
from binance_client import get_klines, get_ticker_price
from indicators  import calc_rsi, rsi_signal, volume_spike, volume_ratio, candle_confirmation
from market_structure import analyse_structure, detect_liquidity_sweep
from zones import get_active_zones

logger = logging.getLogger(__name__)


# ── Signal output ─────────────────────────────────────────────

@dataclass
class TradeSignal:
    pair:           str
    direction:      str         # 'LONG' | 'SHORT'
    entry:          float
    stop_loss:      float
    tp1:            float
    tp2:            float
    rr_ratio:       float
    score:          int         # 0–100
    risk_level:     str         # Low / Medium / High
    confirmations:  list  = field(default_factory=list)
    htf_trend:      str   = ""
    ltf_structure:  str   = ""
    liquidity_desc: str   = ""
    zone_desc:      str   = ""
    rsi_val:        float = 0.0
    vol_ratio:      float = 0.0
    candle_pattern: str   = ""
    valid:          bool  = True
    reason:         str   = ""   # why signal was rejected


# ── Scoring ───────────────────────────────────────────────────

def _score_signal(
    htf_aligned: bool,
    ltf_bos: bool,
    ltf_choch: bool,
    liq_sweep: bool,
    in_ob: bool,
    in_fvg: bool,
    rsi_ok: bool,
    vol_spike: bool,
    candle_ok: bool,
) -> int:
    score = 0
    if htf_aligned:  score += 20
    if ltf_bos:      score += 10
    if ltf_choch:    score += 12
    if liq_sweep:    score += 15
    if in_ob:        score += 13
    if in_fvg:       score += 10
    if rsi_ok:       score += 10
    if vol_spike:    score += 5
    if candle_ok:    score += 5
    return min(score, 100)


def _risk_level(score: int) -> str:
    if score >= 75:
        return "Low"
    if score >= 60:
        return "Medium"
    return "High"


# ── SL / TP calculation ───────────────────────────────────────

def _calc_long_levels(entry: float, structure_low: float
                      ) -> tuple[float, float, float]:
    sl  = structure_low * (1 - SL_BUFFER_PCT)
    risk = entry - sl
    tp1 = entry + risk * 1.5
    tp2 = entry + risk * MIN_RR
    return sl, tp1, tp2


def _calc_short_levels(entry: float, structure_high: float
                       ) -> tuple[float, float, float]:
    sl   = structure_high * (1 + SL_BUFFER_PCT)
    risk = sl - entry
    tp1  = entry - risk * 1.5
    tp2  = entry - risk * MIN_RR
    return sl, tp1, tp2


def _rr(entry, sl, tp2):
    risk   = abs(entry - sl)
    reward = abs(tp2 - entry)
    return round(reward / risk, 2) if risk > 0 else 0.0


# ── Main analysis ─────────────────────────────────────────────

def analyse_pair(symbol: str) -> Optional[TradeSignal]:
    """
    Full SMC analysis for one symbol.
    Returns a TradeSignal or None if no valid setup.
    """
    try:
        htf_df = get_klines(symbol, HTF, HTF_LIMIT)
        ltf_df = get_klines(symbol, LTF, LTF_LIMIT)
        price  = get_ticker_price(symbol)
    except Exception as exc:
        logger.error("Data fetch failed for %s: %s", symbol, exc)
        return None

    # ── HTF trend ──────────────────────────────────────────────
    htf_ms   = analyse_structure(htf_df)
    htf_trend = htf_ms.trend          # 'bullish' | 'bearish' | 'ranging'

    if htf_trend == "ranging":
        return None     # skip ranging market

    # ── LTF structure ──────────────────────────────────────────
    ltf_ms = analyse_structure(ltf_df)

    # ── Determine candidate direction ─────────────────────────
    # Primary direction aligns with HTF trend
    direction = "long" if htf_trend == "bullish" else "short"

    # CHOCH on LTF can override (potential reversal entry)
    if ltf_ms.last_choch == "bullish_choch" and htf_trend == "bullish":
        direction = "long"
    elif ltf_ms.last_choch == "bearish_choch" and htf_trend == "bearish":
        direction = "short"

    htf_aligned = (
        (direction == "long"  and htf_trend == "bullish") or
        (direction == "short" and htf_trend == "bearish")
    )

    # ── Liquidity sweep (LTF) ──────────────────────────────────
    liq = detect_liquidity_sweep(ltf_df,
                                 ltf_ms.swing_highs,
                                 ltf_ms.swing_lows)
    liq_match = (
        (direction == "long"  and liq.direction == "low_sweep") or
        (direction == "short" and liq.direction == "high_sweep")
    )

    # ── Order Blocks & FVG (LTF) ──────────────────────────────
    zones = get_active_zones(ltf_df, direction, price)

    # ── RSI (LTF) ─────────────────────────────────────────────
    rsi_series = calc_rsi(ltf_df["close"])
    rsi_val    = round(rsi_series.iloc[-1], 1)
    rsi_st     = rsi_signal(rsi_val)
    rsi_ok     = (direction == "long"  and rsi_st == "oversold") or \
                 (direction == "short" and rsi_st == "overbought")

    # ── Volume (LTF) ──────────────────────────────────────────
    vol_spike_flag = volume_spike(ltf_df)
    vol_r          = volume_ratio(ltf_df)

    # ── Candle pattern (LTF) ──────────────────────────────────
    candle_ok, candle_desc = candle_confirmation(ltf_df, direction)

    # ── Score ─────────────────────────────────────────────────
    ltf_bos   = ltf_ms.last_bos   is not None
    ltf_choch = ltf_ms.last_choch is not None

    score = _score_signal(
        htf_aligned, ltf_bos, ltf_choch,
        liq_match,
        zones.in_ob, zones.in_fvg,
        rsi_ok, vol_spike_flag, candle_ok,
    )

    if score < MIN_SCORE:
        return None     # not high-probability enough

    # ── Risk management ────────────────────────────────────────
    swing_highs = ltf_ms.swing_highs
    swing_lows  = ltf_ms.swing_lows

    if direction == "long":
        structure_low = swing_lows[-1].price if swing_lows else price * 0.98
        sl, tp1, tp2  = _calc_long_levels(price, structure_low)
    else:
        structure_high = swing_highs[-1].price if swing_highs else price * 1.02
        sl, tp1, tp2   = _calc_short_levels(price, structure_high)

    rr = _rr(price, sl, tp2)
    if rr < MIN_RR:
        return None     # bad risk-to-reward

    # ── Build confirmations list ───────────────────────────────
    confirmations = []

    confirmations.append(
        f"📊 HTF ({HTF}) trend: {htf_trend.upper()} "
        f"| {htf_ms.description}"
    )
    confirmations.append(
        f"🏗 LTF ({LTF}) structure: {ltf_ms.description}"
    )
    confirmations.append(
        f"💧 Liquidity: {liq.description}"
    )
    confirmations.append(
        f"🧱 Zones: {zones.description}"
    )
    confirmations.append(
        f"📈 RSI({HTF}): {rsi_val} → {rsi_st.upper()}"
    )
    confirmations.append(
        f"📦 Volume: {vol_r}× avg"
        + (" ✅ SPIKE" if vol_spike_flag else "")
    )
    confirmations.append(
        f"🕯 Candle: {candle_desc}"
        + (" ✅" if candle_ok else "")
    )

    return TradeSignal(
        pair        = symbol,
        direction   = direction.upper(),
        entry       = round(price, 6),
        stop_loss   = round(sl,  6),
        tp1         = round(tp1, 6),
        tp2         = round(tp2, 6),
        rr_ratio    = rr,
        score       = score,
        risk_level  = _risk_level(score),
        confirmations = confirmations,
        htf_trend   = htf_trend,
        ltf_structure = ltf_ms.description,
        liquidity_desc = liq.description,
        zone_desc   = zones.description,
        rsi_val     = rsi_val,
        vol_ratio   = vol_r,
        candle_pattern = candle_desc,
    )
