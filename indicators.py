# ============================================================
#  indicators.py  –  RSI, volume, and candle-pattern helpers
# ============================================================

import numpy as np
import pandas as pd
from config import (RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
                    VOLUME_SPIKE_MULT, VOLUME_MA_PERIOD)


# ── RSI ──────────────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_signal(rsi_val: float) -> str:
    """Return 'oversold', 'overbought', or 'neutral'."""
    if rsi_val <= RSI_OVERSOLD:
        return "oversold"
    if rsi_val >= RSI_OVERBOUGHT:
        return "overbought"
    return "neutral"


# ── Volume ───────────────────────────────────────────────────

def volume_spike(df: pd.DataFrame) -> bool:
    """True if the last closed candle's volume is a spike."""
    avg = df["volume"].iloc[-VOLUME_MA_PERIOD - 1:-1].mean()
    last_vol = df["volume"].iloc[-2]          # last *closed* candle
    return last_vol >= avg * VOLUME_SPIKE_MULT


def volume_ratio(df: pd.DataFrame) -> float:
    avg = df["volume"].iloc[-VOLUME_MA_PERIOD - 1:-1].mean()
    last_vol = df["volume"].iloc[-2]
    return round(last_vol / avg, 2) if avg > 0 else 0.0


# ── Candle patterns ──────────────────────────────────────────

def _body(candle: pd.Series) -> float:
    return abs(candle["close"] - candle["open"])


def _full_range(candle: pd.Series) -> float:
    return candle["high"] - candle["low"]


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """Last two closed candles form a bullish engulfing."""
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    return (
        prev["close"] < prev["open"]           # prev is bearish
        and curr["close"] > curr["open"]        # curr is bullish
        and curr["open"]  <= prev["close"]
        and curr["close"] >= prev["open"]
        and _body(curr) >= _body(prev) * 0.8
    )


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    return (
        prev["close"] > prev["open"]
        and curr["close"] < curr["open"]
        and curr["open"]  >= prev["close"]
        and curr["close"] <= prev["open"]
        and _body(curr) >= _body(prev) * 0.8
    )


def is_bullish_rejection(df: pd.DataFrame) -> bool:
    """Hammer / pin bar with long lower wick (bullish rejection)."""
    c = df.iloc[-2]
    rng = _full_range(c)
    if rng == 0:
        return False
    lower_wick = min(c["open"], c["close"]) - c["low"]
    body       = _body(c)
    return lower_wick >= rng * 0.55 and body <= rng * 0.35


def is_bearish_rejection(df: pd.DataFrame) -> bool:
    """Shooting star with long upper wick (bearish rejection)."""
    c = df.iloc[-2]
    rng = _full_range(c)
    if rng == 0:
        return False
    upper_wick = c["high"] - max(c["open"], c["close"])
    body       = _body(c)
    return upper_wick >= rng * 0.55 and body <= rng * 0.35


def candle_confirmation(df: pd.DataFrame, direction: str) -> tuple[bool, str]:
    """
    Return (confirmed, description) for a given direction ('long'/'short').
    """
    if direction == "long":
        if is_bullish_engulfing(df):
            return True, "Bullish Engulfing"
        if is_bullish_rejection(df):
            return True, "Bullish Pin Bar / Hammer"
    elif direction == "short":
        if is_bearish_engulfing(df):
            return True, "Bearish Engulfing"
        if is_bearish_rejection(df):
            return True, "Shooting Star / Rejection"
    return False, "No candle pattern"
