# ============================================================
#  zones.py  –  Order Block & Fair Value Gap detection
# ============================================================

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from config import OB_LOOKBACK, FVG_MIN_GAP_PCT


# ── Data classes ─────────────────────────────────────────────

@dataclass
class OrderBlock:
    kind:   str         # 'bullish' | 'bearish'
    top:    float
    bottom: float
    index:  int
    description: str = ""


@dataclass
class FairValueGap:
    kind:   str         # 'bullish' | 'bearish'
    top:    float
    bottom: float
    index:  int         # middle candle index
    description: str = ""


@dataclass
class ZoneResult:
    order_block:    Optional[OrderBlock]
    fvg:            Optional[FairValueGap]
    in_ob:          bool = False
    in_fvg:         bool = False
    description:    str  = ""


# ── Order Block detection ─────────────────────────────────────

def find_order_blocks(df: pd.DataFrame,
                      direction: str,
                      lookback: int = OB_LOOKBACK) -> list[OrderBlock]:
    """
    Bullish OB: last bearish candle before a strong, consecutive bullish move.
    Bearish OB: last bullish candle before a strong, consecutive bearish move.
    """
    blocks = []
    subset = df.iloc[-(lookback + 5):-1]   # recent closed candles only

    for i in range(2, len(subset) - 2):
        c0 = subset.iloc[i]
        c1 = subset.iloc[i + 1]
        c2 = subset.iloc[i + 2]

        if direction == "long":
            # Last bearish candle followed by 2+ bullish candles = bullish OB
            if (c0["close"] < c0["open"]
                    and c1["close"] > c1["open"]
                    and c2["close"] > c2["open"]
                    and c2["close"] > c0["open"]):    # strong move
                blocks.append(OrderBlock(
                    kind="bullish",
                    top=c0["open"],
                    bottom=c0["low"],
                    index=i,
                    description=(
                        f"Bullish OB zone "
                        f"{c0['low']:.4f} – {c0['open']:.4f}"
                    ),
                ))

        elif direction == "short":
            # Last bullish candle followed by 2+ bearish candles = bearish OB
            if (c0["close"] > c0["open"]
                    and c1["close"] < c1["open"]
                    and c2["close"] < c2["open"]
                    and c2["close"] < c0["open"]):
                blocks.append(OrderBlock(
                    kind="bearish",
                    top=c0["high"],
                    bottom=c0["open"],
                    index=i,
                    description=(
                        f"Bearish OB zone "
                        f"{c0['open']:.4f} – {c0['high']:.4f}"
                    ),
                ))

    return blocks


# ── Fair Value Gap detection ──────────────────────────────────

def find_fvgs(df: pd.DataFrame,
              direction: str,
              min_gap_pct: float = FVG_MIN_GAP_PCT) -> list[FairValueGap]:
    """
    Bullish FVG: candle[0].low > candle[2].high  (gap above candle[2])
    Bearish FVG: candle[0].high < candle[2].low  (gap below candle[0])
    """
    fvgs   = []
    subset = df.iloc[-50:-1]

    for i in range(len(subset) - 2):
        c0 = subset.iloc[i]
        c2 = subset.iloc[i + 2]
        mid_price = (c0["close"] + c2["open"]) / 2

        if direction == "long":
            # Price gap between c0 low and c2 high (imbalance, bullish)
            if c0["low"] > c2["high"]:
                gap  = c0["low"] - c2["high"]
                pct  = gap / mid_price * 100
                if pct >= min_gap_pct:
                    fvgs.append(FairValueGap(
                        kind="bullish",
                        top=c0["low"],
                        bottom=c2["high"],
                        index=i + 1,
                        description=(
                            f"Bullish FVG {c2['high']:.4f} – {c0['low']:.4f} "
                            f"({pct:.2f}% gap)"
                        ),
                    ))

        elif direction == "short":
            # Gap between c2 low and c0 high (bearish imbalance)
            if c2["low"] > c0["high"]:
                gap  = c2["low"] - c0["high"]
                pct  = gap / mid_price * 100
                if pct >= min_gap_pct:
                    fvgs.append(FairValueGap(
                        kind="bearish",
                        top=c2["low"],
                        bottom=c0["high"],
                        index=i + 1,
                        description=(
                            f"Bearish FVG {c0['high']:.4f} – {c2['low']:.4f} "
                            f"({pct:.2f}% gap)"
                        ),
                    ))

    return fvgs


# ── Check if price is inside a zone ──────────────────────────

def price_in_zone(price: float, top: float, bottom: float,
                  tolerance: float = 0.002) -> bool:
    """True if price is within the zone (with a small tolerance buffer)."""
    return (bottom * (1 - tolerance)) <= price <= (top * (1 + tolerance))


# ── Combined zone lookup ──────────────────────────────────────

def get_active_zones(df: pd.DataFrame,
                     direction: str,
                     current_price: float) -> ZoneResult:
    obs  = find_order_blocks(df, direction)
    fvgs = find_fvgs(df, direction)

    result = ZoneResult(order_block=None, fvg=None)
    desc_parts = []

    # Most-recent order block
    if obs:
        ob = obs[-1]
        result.order_block = ob
        result.in_ob = price_in_zone(current_price, ob.top, ob.bottom)
        flag = " ✅ PRICE IN ZONE" if result.in_ob else ""
        desc_parts.append(ob.description + flag)

    # Most-recent FVG
    if fvgs:
        fvg = fvgs[-1]
        result.fvg = fvg
        result.in_fvg = price_in_zone(current_price, fvg.top, fvg.bottom)
        flag = " ✅ PRICE IN FVG" if result.in_fvg else ""
        desc_parts.append(fvg.description + flag)

    if not desc_parts:
        desc_parts.append("No OB / FVG detected near current price")

    result.description = " | ".join(desc_parts)
    return result
