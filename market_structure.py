# ============================================================
#  market_structure.py  –  BOS / CHOCH / HH-HL / Liquidity
# ============================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from config import SWING_LOOKBACK


# ── Data classes ─────────────────────────────────────────────

@dataclass
class SwingPoint:
    index: int
    price: float
    kind:  str          # 'high' or 'low'


@dataclass
class MarketStructure:
    trend:          str             # 'bullish', 'bearish', 'ranging'
    last_bos:       Optional[str]   # 'bullish_bos' | 'bearish_bos' | None
    last_choch:     Optional[str]   # 'bullish_choch' | 'bearish_choch' | None
    swing_highs:    list = field(default_factory=list)
    swing_lows:     list = field(default_factory=list)
    hh_hl:          bool = False    # higher highs + higher lows
    lh_ll:          bool = False    # lower highs + lower lows
    description:    str = ""


@dataclass
class LiquidityEvent:
    detected:   bool
    direction:  str     # 'high_sweep' | 'low_sweep' | 'none'
    level:      float
    description: str


# ── Swing pivot detection ────────────────────────────────────

def find_swing_points(df: pd.DataFrame,
                      lookback: int = SWING_LOOKBACK) -> tuple[list, list]:
    """
    Return (swing_highs, swing_lows) as lists of SwingPoint.
    A swing high at index i means df.high[i] is the highest within
    [i-lookback .. i+lookback].
    """
    highs, lows = [], []
    n = len(df)

    for i in range(lookback, n - lookback):
        window_h = df["high"].iloc[i - lookback: i + lookback + 1]
        window_l = df["low"].iloc[i  - lookback: i + lookback + 1]

        if df["high"].iloc[i] == window_h.max():
            highs.append(SwingPoint(i, df["high"].iloc[i], "high"))
        if df["low"].iloc[i] == window_l.min():
            lows.append(SwingPoint(i, df["low"].iloc[i], "low"))

    return highs, lows


# ── Market structure analysis ────────────────────────────────

def analyse_structure(df: pd.DataFrame) -> MarketStructure:
    swing_highs, swing_lows = find_swing_points(df)

    ms = MarketStructure(
        trend="ranging",
        last_bos=None,
        last_choch=None,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
    )

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        ms.description = "Insufficient swing data"
        return ms

    # Last two swing highs and lows
    sh = swing_highs[-2:]
    sl = swing_lows[-2:]

    # Higher highs + higher lows → bullish
    hh = sh[1].price > sh[0].price
    hl = sl[1].price > sl[0].price
    # Lower highs + lower lows → bearish
    lh = sh[1].price < sh[0].price
    ll = sl[1].price < sl[0].price

    ms.hh_hl = hh and hl
    ms.lh_ll = lh and ll

    current_close = df["close"].iloc[-1]

    # BOS – price closes beyond last swing level
    if current_close > sh[-1].price:
        ms.last_bos = "bullish_bos"
    elif current_close < sl[-1].price:
        ms.last_bos = "bearish_bos"

    # CHOCH – structure flip
    if ms.lh_ll and current_close > sh[-1].price:
        ms.last_choch = "bullish_choch"
    elif ms.hh_hl and current_close < sl[-1].price:
        ms.last_choch = "bearish_choch"

    # Overall trend
    if ms.hh_hl or ms.last_bos == "bullish_bos":
        ms.trend = "bullish"
    elif ms.lh_ll or ms.last_bos == "bearish_bos":
        ms.trend = "bearish"

    # Human-readable description
    parts = []
    if ms.hh_hl:
        parts.append("HH+HL (uptrend)")
    if ms.lh_ll:
        parts.append("LH+LL (downtrend)")
    if ms.last_bos:
        parts.append(ms.last_bos.replace("_", " ").upper())
    if ms.last_choch:
        parts.append(f"CHOCH → {ms.last_choch.split('_')[0].upper()}")
    ms.description = " | ".join(parts) if parts else "Ranging / no clear structure"

    return ms


# ── Liquidity sweep detection ─────────────────────────────────

def detect_liquidity_sweep(df: pd.DataFrame,
                            swing_highs: list,
                            swing_lows: list,
                            lookback: int = 3) -> LiquidityEvent:
    """
    A liquidity sweep occurs when the current candle wicks beyond a
    recent swing high/low but closes back on the other side.
    """
    if not swing_highs or not swing_lows:
        return LiquidityEvent(False, "none", 0.0, "No swing data")

    # Focus on recent candles
    recent = df.iloc[-lookback - 1:-1]   # last N *closed* candles

    # Look for sweep above recent swing highs
    for sh in reversed(swing_highs[-5:]):
        for _, row in recent.iterrows():
            if row["high"] > sh.price and row["close"] < sh.price:
                return LiquidityEvent(
                    detected=True,
                    direction="high_sweep",
                    level=sh.price,
                    description=(
                        f"Liquidity sweep ABOVE swing high at "
                        f"{sh.price:.4f} — stop-hunt detected (bearish)"
                    ),
                )

    # Look for sweep below recent swing lows
    for sl in reversed(swing_lows[-5:]):
        for _, row in recent.iterrows():
            if row["low"] < sl.price and row["close"] > sl.price:
                return LiquidityEvent(
                    detected=True,
                    direction="low_sweep",
                    level=sl.price,
                    description=(
                        f"Liquidity sweep BELOW swing low at "
                        f"{sl.price:.4f} — stop-hunt detected (bullish)"
                    ),
                )

    return LiquidityEvent(False, "none", 0.0, "No liquidity sweep detected")
