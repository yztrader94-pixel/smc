#!/usr/bin/env python3
# ============================================================
#  bot_ccxt.py  –  SMC Signal Bot using ccxt (no API key)
#  Works with Binance Futures public data via ccxt library.
#  Features: BOS, CHOCH, Order Blocks, FVG, Liquidity Sweeps,
#            RSI, Volume, Candle Patterns, Live TP/SL tracking
# ============================================================

import asyncio
import logging
import sys
import warnings
from collections import deque
from datetime import datetime, timedelta

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
#  CONFIG — edit only this section
# ════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = "7732870721:AAEHG3QJdo31S9sA8xjJzf-cXj6Tn4mo2uo"
TELEGRAM_CHAT_ID = "7500072234"

HTF              = "4h"
LTF              = "15m"
SCAN_INTERVAL_MIN = 15
MIN_VOLUME_USDT  = 5_000_000    # $5M min 24h volume filter
MAX_CONCURRENT   = 8            # parallel pair analyses
MIN_SCORE        = 55           # minimum signal score (0–100)
MIN_RR           = 2.0          # minimum risk:reward

RSI_PERIOD       = 14
RSI_OVERSOLD     = 35
RSI_OVERBOUGHT   = 65
VOLUME_SPIKE_MULT = 1.6
SWING_LOOKBACK   = 10
SL_BUFFER_PCT    = 0.002

# ════════════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════════════

def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def volume_spike(df: pd.DataFrame, mult: float = VOLUME_SPIKE_MULT) -> tuple:
    avg = df["volume"].iloc[-21:-1].mean()
    last = df["volume"].iloc[-2]
    ratio = round(last / avg, 2) if avg > 0 else 0
    return last >= avg * mult, ratio


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    p, c = df.iloc[-3], df.iloc[-2]
    return (p["close"] < p["open"] and c["close"] > c["open"]
            and c["open"] <= p["close"] and c["close"] >= p["open"])


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    p, c = df.iloc[-3], df.iloc[-2]
    return (p["close"] > p["open"] and c["close"] < c["open"]
            and c["open"] >= p["close"] and c["close"] <= p["open"])


def is_bullish_pin(df: pd.DataFrame) -> bool:
    c = df.iloc[-2]
    rng = c["high"] - c["low"]
    if rng == 0: return False
    lower = min(c["open"], c["close"]) - c["low"]
    return lower >= rng * 0.55 and abs(c["close"] - c["open"]) <= rng * 0.35


def is_bearish_pin(df: pd.DataFrame) -> bool:
    c = df.iloc[-2]
    rng = c["high"] - c["low"]
    if rng == 0: return False
    upper = c["high"] - max(c["open"], c["close"])
    return upper >= rng * 0.55 and abs(c["close"] - c["open"]) <= rng * 0.35


# ════════════════════════════════════════════════════════════
#  MARKET STRUCTURE
# ════════════════════════════════════════════════════════════

def find_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK):
    highs, lows = [], []
    n = len(df)
    for i in range(lookback, n - lookback):
        wh = df["high"].iloc[i - lookback: i + lookback + 1]
        wl = df["low"].iloc[i  - lookback: i + lookback + 1]
        if df["high"].iloc[i] == wh.max():
            highs.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == wl.min():
            lows.append((i, df["low"].iloc[i]))
    return highs, lows


def analyse_structure(df: pd.DataFrame) -> dict:
    sh, sl = find_swings(df)
    result = {
        "trend": "ranging", "bos": None, "choch": None,
        "hh_hl": False, "lh_ll": False, "description": "",
        "swing_highs": sh, "swing_lows": sl,
    }
    if len(sh) < 2 or len(sl) < 2:
        result["description"] = "Insufficient swing data"
        return result

    hh = sh[-1][1] > sh[-2][1]
    hl = sl[-1][1] > sl[-2][1]
    lh = sh[-1][1] < sh[-2][1]
    ll = sl[-1][1] < sl[-2][1]
    result["hh_hl"] = hh and hl
    result["lh_ll"] = lh and ll

    close = df["close"].iloc[-1]
    if close > sh[-1][1]:
        result["bos"] = "bullish"
    elif close < sl[-1][1]:
        result["bos"] = "bearish"

    if result["lh_ll"] and close > sh[-1][1]:
        result["choch"] = "bullish"
    elif result["hh_hl"] and close < sl[-1][1]:
        result["choch"] = "bearish"

    if result["hh_hl"] or result["bos"] == "bullish":
        result["trend"] = "bullish"
    elif result["lh_ll"] or result["bos"] == "bearish":
        result["trend"] = "bearish"

    parts = []
    if result["hh_hl"]:  parts.append("HH+HL ✅")
    if result["lh_ll"]:  parts.append("LH+LL ✅")
    if result["bos"]:    parts.append(f"BOS {result['bos'].upper()}")
    if result["choch"]:  parts.append(f"CHOCH → {result['choch'].upper()}")
    result["description"] = " | ".join(parts) or "Ranging"
    return result


def detect_liquidity_sweep(df: pd.DataFrame, sh: list, sl: list) -> dict:
    result = {"detected": False, "direction": "none", "level": 0, "description": "No sweep"}
    recent = df.iloc[-4:-1]
    for _, hp in reversed(sh[-5:]):
        for _, row in recent.iterrows():
            if row["high"] > hp and row["close"] < hp:
                return {"detected": True, "direction": "high_sweep",
                        "level": hp, "description": f"Sweep ABOVE {hp:.4f} (bearish stop-hunt)"}
    for _, lp in reversed(sl[-5:]):
        for _, row in recent.iterrows():
            if row["low"] < lp and row["close"] > lp:
                return {"detected": True, "direction": "low_sweep",
                        "level": lp, "description": f"Sweep BELOW {lp:.4f} (bullish stop-hunt)"}
    return result


# ════════════════════════════════════════════════════════════
#  ORDER BLOCKS & FVG
# ════════════════════════════════════════════════════════════

def find_order_block(df: pd.DataFrame, direction: str) -> dict:
    subset = df.iloc[-31:-1]
    for i in range(2, len(subset) - 2):
        c0, c1, c2 = subset.iloc[i], subset.iloc[i+1], subset.iloc[i+2]
        if direction == "long":
            if (c0["close"] < c0["open"] and c1["close"] > c1["open"]
                    and c2["close"] > c2["open"] and c2["close"] > c0["open"]):
                return {"found": True, "top": c0["open"], "bottom": c0["low"],
                        "description": f"Bullish OB {c0['low']:.4f}–{c0['open']:.4f}"}
        elif direction == "short":
            if (c0["close"] > c0["open"] and c1["close"] < c1["open"]
                    and c2["close"] < c2["open"] and c2["close"] < c0["open"]):
                return {"found": True, "top": c0["high"], "bottom": c0["open"],
                        "description": f"Bearish OB {c0['open']:.4f}–{c0['high']:.4f}"}
    return {"found": False, "top": 0, "bottom": 0, "description": "No OB"}


def find_fvg(df: pd.DataFrame, direction: str) -> dict:
    subset = df.iloc[-50:-1]
    for i in range(len(subset) - 2):
        c0, c2 = subset.iloc[i], subset.iloc[i+2]
        mid = (c0["close"] + c2["open"]) / 2
        if direction == "long" and c0["low"] > c2["high"]:
            gap_pct = (c0["low"] - c2["high"]) / mid * 100
            if gap_pct >= 0.08:
                return {"found": True, "top": c0["low"], "bottom": c2["high"],
                        "description": f"Bullish FVG {c2['high']:.4f}–{c0['low']:.4f} ({gap_pct:.2f}%)"}
        elif direction == "short" and c2["low"] > c0["high"]:
            gap_pct = (c2["low"] - c0["high"]) / mid * 100
            if gap_pct >= 0.08:
                return {"found": True, "top": c2["low"], "bottom": c0["high"],
                        "description": f"Bearish FVG {c0['high']:.4f}–{c2['low']:.4f} ({gap_pct:.2f}%)"}
    return {"found": False, "top": 0, "bottom": 0, "description": "No FVG"}


def price_in_zone(price, top, bottom, tol=0.003):
    return bottom * (1 - tol) <= price <= top * (1 + tol)


# ════════════════════════════════════════════════════════════
#  SCORING
# ════════════════════════════════════════════════════════════

def score_signal(htf_aligned, ltf_bos, ltf_choch, liq_match,
                 in_ob, in_fvg, rsi_ok, vol_spike_ok, candle_ok) -> int:
    s = 0
    if htf_aligned: s += 20
    if ltf_bos:     s += 10
    if ltf_choch:   s += 12
    if liq_match:   s += 15
    if in_ob:       s += 13
    if in_fvg:      s += 10
    if rsi_ok:      s += 10
    if vol_spike_ok: s += 5
    if candle_ok:   s += 5
    return min(s, 100)


def risk_level(score: int) -> str:
    return "Low 🟢" if score >= 75 else "Medium 🟡" if score >= 60 else "High 🔴"


def quality_label(score: int) -> str:
    return "PREMIUM 💎" if score >= 75 else "HIGH 🔥" if score >= 60 else "GOOD ✅"


# ════════════════════════════════════════════════════════════
#  SIGNAL FORMATTER
# ════════════════════════════════════════════════════════════

def bar(score: int, n: int = 10) -> str:
    f = round(score / 100 * n)
    return "█" * f + "░" * (n - f)


def format_signal(sig: dict) -> str:
    e = "🟢" if sig["direction"] == "LONG" else "🔴"
    lines = [
        f"{'='*38}",
        f"{e} <b>{sig['pair']}  —  {sig['direction']}</b>  {quality_label(sig['score'])}",
        f"{'='*38}",
        "",
        f"💰 <b>ENTRY:</b>      <code>{sig['entry']}</code>",
        f"🛑 <b>STOP LOSS:</b>  <code>{sig['sl']}</code>",
        f"🎯 <b>TP 1:</b>       <code>{sig['tp1']}</code>",
        f"🚀 <b>TP 2:</b>       <code>{sig['tp2']}</code>",
        f"⚖️  <b>R:R:</b>        <code>1 : {sig['rr']}</code>",
        "",
        f"📊 <b>Score:</b>  {sig['score']}%  {bar(sig['score'])}",
        f"⚠️  <b>Risk:</b>   {risk_level(sig['score'])}",
        "",
        "─"*36,
        "<b>📋 CONFIRMATIONS</b>",
        "─"*36,
        f"  • 📈 HTF ({HTF}) trend: <b>{sig['htf_trend'].upper()}</b>",
        f"  • 🏗 LTF ({LTF}) structure: {sig['ltf_desc']}",
        f"  • 💧 Liquidity: {sig['liq_desc']}",
        f"  • 🧱 Order Block: {sig['ob_desc']}",
        f"  • 🌀 Fair Value Gap: {sig['fvg_desc']}",
        f"  • 📉 RSI: <b>{sig['rsi']}</b> → {sig['rsi_state']}",
        f"  • 📦 Volume: <b>{sig['vol_ratio']}×</b> avg{' ✅ SPIKE' if sig['vol_spike'] else ''}",
        f"  • 🕯 Candle: {sig['candle']}",
        "",
        "─"*36,
        f"<b>📡 LIVE TRACKING ACTIVE</b>",
        f"<i>⏰ {sig['time']}</i>",
        f"{'='*38}",
        "",
        "<i>⚠️ Not financial advice. Manage risk.</i>",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  MAIN SCANNER CLASS
# ════════════════════════════════════════════════════════════

class SMCScanner:
    def __init__(self):
        self.exchange = ccxt.binance({
            "apiKey": None,
            "secret": None,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.signal_history = deque(maxlen=200)
        self.active_trades: dict = {}
        self.is_scanning = False
        self.stats = {
            "total": 0, "long": 0, "short": 0, "premium": 0,
            "tp1": 0, "tp2": 0, "sl_hits": 0,
            "last_scan": None, "pairs_scanned": 0,
        }

    # ── Exchange helpers ──────────────────────────────────────

    async def get_all_pairs(self) -> list:
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = []
            for sym in self.exchange.symbols:
                if not sym.endswith("/USDT:USDT"):
                    continue
                t = tickers.get(sym, {})
                if t.get("quoteVolume", 0) >= MIN_VOLUME_USDT:
                    pairs.append(sym)
            pairs.sort(key=lambda s: tickers.get(s, {}).get("quoteVolume", 0), reverse=True)
            logger.info("Found %d liquid USDT pairs.", len(pairs))
            return pairs
        except Exception as exc:
            logger.error("get_all_pairs failed: %s", exc)
            return []

    async def fetch_ohlcv(self, symbol: str, tf: str, limit: int = 200) -> pd.DataFrame:
        raw = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df

    # ── Analysis core ─────────────────────────────────────────

    def analyse_pair(self, symbol: str, htf_df: pd.DataFrame,
                     ltf_df: pd.DataFrame) -> dict | None:
        price = ltf_df["close"].iloc[-1]

        # HTF structure
        htf_ms = analyse_structure(htf_df)
        if htf_ms["trend"] == "ranging":
            return None

        direction = "long" if htf_ms["trend"] == "bullish" else "short"
        htf_aligned = True   # by definition since we picked direction from HTF

        # LTF structure
        ltf_ms = analyse_structure(ltf_df)

        # Liquidity sweep
        liq = detect_liquidity_sweep(ltf_df, ltf_ms["swing_highs"], ltf_ms["swing_lows"])
        liq_match = (
            (direction == "long"  and liq["direction"] == "low_sweep") or
            (direction == "short" and liq["direction"] == "high_sweep")
        )

        # Order Block & FVG
        ob  = find_order_block(ltf_df, direction)
        fvg = find_fvg(ltf_df, direction)
        in_ob  = ob["found"]  and price_in_zone(price, ob["top"],  ob["bottom"])
        in_fvg = fvg["found"] and price_in_zone(price, fvg["top"], fvg["bottom"])

        # RSI
        rsi_s = calc_rsi(ltf_df["close"])
        rsi_v = round(rsi_s.iloc[-1], 1)
        rsi_state = ("OVERSOLD"   if rsi_v <= RSI_OVERSOLD
                     else "OVERBOUGHT" if rsi_v >= RSI_OVERBOUGHT
                     else "neutral")
        rsi_ok = (direction == "long"  and rsi_state == "OVERSOLD") or \
                 (direction == "short" and rsi_state == "OVERBOUGHT")

        # Volume
        vol_spike_ok, vol_r = volume_spike(ltf_df)

        # Candle pattern
        if direction == "long":
            candle_ok = is_bullish_engulfing(ltf_df) or is_bullish_pin(ltf_df)
            candle_desc = ("Bullish Engulfing ✅" if is_bullish_engulfing(ltf_df)
                           else "Bullish Pin Bar ✅" if is_bullish_pin(ltf_df)
                           else "No pattern")
        else:
            candle_ok = is_bearish_engulfing(ltf_df) or is_bearish_pin(ltf_df)
            candle_desc = ("Bearish Engulfing ✅" if is_bearish_engulfing(ltf_df)
                           else "Shooting Star ✅" if is_bearish_pin(ltf_df)
                           else "No pattern")

        # Score
        sc = score_signal(
            htf_aligned,
            ltf_ms["bos"] is not None,
            ltf_ms["choch"] is not None,
            liq_match, in_ob, in_fvg,
            rsi_ok, vol_spike_ok, candle_ok,
        )
        if sc < MIN_SCORE:
            return None

        # SL / TP
        sh_list = ltf_ms["swing_highs"]
        sl_list = ltf_ms["swing_lows"]
        if direction == "long":
            struct_low = sl_list[-1][1] if sl_list else price * 0.98
            sl  = struct_low * (1 - SL_BUFFER_PCT)
            risk = price - sl
            tp1 = price + risk * 1.5
            tp2 = price + risk * MIN_RR
        else:
            struct_high = sh_list[-1][1] if sh_list else price * 1.02
            sl  = struct_high * (1 + SL_BUFFER_PCT)
            risk = sl - price
            tp1 = price - risk * 1.5
            tp2 = price - risk * MIN_RR

        if risk <= 0:
            return None
        rr = round(abs(tp2 - price) / risk, 2)
        if rr < MIN_RR:
            return None

        def fmt(p): return f"{p:.6f}" if p < 1 else f"{p:.4f}" if p < 100 else f"{p:.2f}"

        trade_id = f"{symbol.replace('/USDT:USDT','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return {
            "trade_id":   trade_id,
            "pair":       symbol.replace("/USDT:USDT", ""),
            "full_symbol": symbol,
            "direction":  direction.upper(),
            "entry":      fmt(price),
            "entry_raw":  price,
            "sl":         fmt(sl),
            "sl_raw":     sl,
            "tp1":        fmt(tp1),
            "tp1_raw":    tp1,
            "tp2":        fmt(tp2),
            "tp2_raw":    tp2,
            "rr":         rr,
            "score":      sc,
            "htf_trend":  htf_ms["trend"],
            "ltf_desc":   ltf_ms["description"],
            "liq_desc":   liq["description"],
            "ob_desc":    ob["description"],
            "fvg_desc":   fvg["description"],
            "rsi":        rsi_v,
            "rsi_state":  rsi_state,
            "vol_ratio":  vol_r,
            "vol_spike":  vol_spike_ok,
            "candle":     candle_desc,
            "time":       datetime.now().strftime("%H:%M:%S"),
            "timestamp":  datetime.now(),
            "tp1_hit":    False,
            "tp2_hit":    False,
            "sl_hit":     False,
            "status":     "ACTIVE",
        }

    # ── Scan all pairs ────────────────────────────────────────

    async def scan_all(self) -> list:
        if self.is_scanning:
            logger.info("Already scanning — skipped.")
            return []

        self.is_scanning = True
        logger.info("▶ Scan started …")
        pairs    = await self.get_all_pairs()
        signals  = []
        scanned  = 0
        sem      = asyncio.Semaphore(MAX_CONCURRENT)

        async def process(sym):
            nonlocal scanned
            async with sem:
                try:
                    htf_df = await self.fetch_ohlcv(sym, HTF, 200)
                    await asyncio.sleep(0.1)
                    ltf_df = await self.fetch_ohlcv(sym, LTF, 200)
                    await asyncio.sleep(0.1)
                    sig = self.analyse_pair(sym, htf_df, ltf_df)
                    scanned += 1
                    if scanned % 20 == 0:
                        logger.info("  Progress: %d / %d", scanned, len(pairs))
                    return sig
                except Exception as exc:
                    logger.warning("Error %s: %s", sym, exc)
                    scanned += 1
                    return None

        tasks   = [process(p) for p in pairs]
        results = await asyncio.gather(*tasks)

        for sig in results:
            if sig is None:
                continue
            signals.append(sig)
            self.signal_history.append(sig)
            self.active_trades[sig["trade_id"]] = sig
            self.stats["total"]  += 1
            self.stats["long"]   += sig["direction"] == "LONG"
            self.stats["short"]  += sig["direction"] == "SHORT"
            self.stats["premium"] += sig["score"] >= 75
            await self.send(format_signal(sig))
            await asyncio.sleep(0.5)

        self.stats["last_scan"]     = datetime.now()
        self.stats["pairs_scanned"] = scanned

        longs    = sum(1 for s in signals if s["direction"] == "LONG")
        shorts   = len(signals) - longs
        premium  = sum(1 for s in signals if s["score"] >= 75)

        summary = (
            f"✅ <b>SCAN COMPLETE</b>\n\n"
            f"📊 Scanned: <b>{scanned}</b>\n"
            f"🎯 Signals: <b>{len(signals)}</b>\n"
            f"  🟢 Long: {longs}\n"
            f"  🔴 Short: {shorts}\n"
            f"  💎 Premium: {premium}\n"
            f"  📡 Tracking: {len(self.active_trades)}\n\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send(summary)
        logger.info("◀ Scan done — %d signals from %d pairs.", len(signals), scanned)
        self.is_scanning = False
        return signals

    # ── Live trade tracker ────────────────────────────────────

    async def track_trades(self):
        logger.info("📡 Trade tracker started.")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30)
                    continue

                to_close = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        # Auto-close after 24h
                        if datetime.now() - t["timestamp"] > timedelta(hours=24):
                            await self.send(
                                f"⏰ <b>24H EXPIRED</b>\n"
                                f"<code>{tid}</code>  {t['pair']}  {t['direction']}\n"
                                f"Close your position manually."
                            )
                            to_close.append(tid)
                            continue

                        ticker = await self.exchange.fetch_ticker(t["full_symbol"])
                        price  = float(ticker["last"])
                        entry  = t["entry_raw"]
                        tp1    = t["tp1_raw"]
                        tp2    = t["tp2_raw"]
                        sl     = t["sl_raw"]

                        if t["direction"] == "LONG":
                            if not t["tp1_hit"] and price >= tp1:
                                t["tp1_hit"] = True
                                self.stats["tp1"] += 1
                                pct = round((tp1 - entry) / entry * 100, 2)
                                await self.send(
                                    f"🎯 <b>TP1 HIT!</b>  {t['pair']}\n"
                                    f"Target: <code>{t['tp1']}</code>  (+{pct}%)\n"
                                    f"📋 Take 50% profit · Move SL to breakeven"
                                )
                            if not t["tp2_hit"] and price >= tp2:
                                t["tp2_hit"] = True
                                self.stats["tp2"] += 1
                                pct = round((tp2 - entry) / entry * 100, 2)
                                await self.send(
                                    f"🚀 <b>TP2 HIT!</b>  {t['pair']}\n"
                                    f"Target: <code>{t['tp2']}</code>  (+{pct}%)\n"
                                    f"🎊 Full target reached — trade complete!"
                                )
                                to_close.append(tid)
                            if not t["sl_hit"] and price <= sl:
                                t["sl_hit"] = True
                                self.stats["sl_hits"] += 1
                                loss = round((entry - sl) / entry * 100, 2)
                                await self.send(
                                    f"⛔ <b>STOP LOSS HIT</b>  {t['pair']}\n"
                                    f"SL: <code>{t['sl']}</code>  (-{loss}%)"
                                )
                                to_close.append(tid)
                        else:  # SHORT
                            if not t["tp1_hit"] and price <= tp1:
                                t["tp1_hit"] = True
                                self.stats["tp1"] += 1
                                pct = round((entry - tp1) / entry * 100, 2)
                                await self.send(
                                    f"🎯 <b>TP1 HIT!</b>  {t['pair']}\n"
                                    f"Target: <code>{t['tp1']}</code>  (+{pct}%)\n"
                                    f"📋 Take 50% profit · Move SL to breakeven"
                                )
                            if not t["tp2_hit"] and price <= tp2:
                                t["tp2_hit"] = True
                                self.stats["tp2"] += 1
                                pct = round((entry - tp2) / entry * 100, 2)
                                await self.send(
                                    f"🚀 <b>TP2 HIT!</b>  {t['pair']}\n"
                                    f"Target: <code>{t['tp2']}</code>  (+{pct}%)\n"
                                    f"🎊 Full target reached — trade complete!"
                                )
                                to_close.append(tid)
                            if not t["sl_hit"] and price >= sl:
                                t["sl_hit"] = True
                                self.stats["sl_hits"] += 1
                                loss = round((sl - entry) / entry * 100, 2)
                                await self.send(
                                    f"⛔ <b>STOP LOSS HIT</b>  {t['pair']}\n"
                                    f"SL: <code>{t['sl']}</code>  (-{loss}%)"
                                )
                                to_close.append(tid)

                        await asyncio.sleep(0.2)

                    except Exception as exc:
                        logger.warning("Tracker error %s: %s", tid, exc)

                for tid in set(to_close):
                    self.active_trades.pop(tid, None)

                await asyncio.sleep(30)

            except Exception as exc:
                logger.error("Tracker loop error: %s", exc)
                await asyncio.sleep(60)

    # ── Telegram send ─────────────────────────────────────────

    async def send(self, text: str):
        try:
            # split if needed
            if len(text) > 4096:
                for i in range(0, len(text), 4000):
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=text[i:i+4000],
                        parse_mode=ParseMode.HTML,
                    )
                    await asyncio.sleep(0.3)
            else:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=text,
                    parse_mode=ParseMode.HTML,
                )
        except Exception as exc:
            logger.error("send failed: %s", exc)

    # ── Main loop ─────────────────────────────────────────────

    async def run(self):
        logger.info("🚀 SMC Signal Bot starting …")
        await self.send(
            "🤖 <b>SMC CRYPTO SIGNAL BOT — ONLINE</b>\n\n"
            "✅ Binance Futures via ccxt (no API key)\n"
            "✅ Multi-timeframe: 4H trend + 15M entry\n"
            "✅ BOS / CHOCH / HH-HL detection\n"
            "✅ Order Blocks + Fair Value Gaps\n"
            "✅ Liquidity sweep detection\n"
            "✅ RSI + Volume + Candle patterns\n"
            "✅ Live TP/SL tracking\n\n"
            f"⏱ Auto-scan every <b>{SCAN_INTERVAL_MIN} min</b>\n\n"
            "<b>Commands:</b> /scan /stats /trades /help"
        )
        asyncio.create_task(self.track_trades())
        while True:
            try:
                await self.scan_all()
                logger.info("💤 Next scan in %d min …", SCAN_INTERVAL_MIN)
                await asyncio.sleep(SCAN_INTERVAL_MIN * 60)
            except Exception as exc:
                logger.error("Main loop error: %s", exc)
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ════════════════════════════════════════════════════════════
#  TELEGRAM COMMAND HANDLERS
# ════════════════════════════════════════════════════════════

class Commands:
    def __init__(self, scanner: SMCScanner):
        self.s = scanner

    async def start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "👋 <b>SMC Signal Bot</b>\n\n"
            "/scan   — Force scan now\n"
            "/stats  — Signal statistics\n"
            "/trades — Active tracked trades\n"
            "/help   — Help & indicator info",
            parse_mode=ParseMode.HTML,
        )

    async def scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await update.message.reply_text("⚠️ Scan already running!")
            return
        await update.message.reply_text("🔍 Scan started…")
        await self.s.scan_all()

    async def stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s = self.s.stats
        last = s["last_scan"].strftime("%H:%M:%S") if s["last_scan"] else "—"
        await update.message.reply_text(
            f"📊 <b>STATISTICS</b>\n\n"
            f"Total signals:  <b>{s['total']}</b>\n"
            f"  🟢 Long:      {s['long']}\n"
            f"  🔴 Short:     {s['short']}\n"
            f"  💎 Premium:   {s['premium']}\n\n"
            f"<b>Results:</b>\n"
            f"  TP1 hits:  {s['tp1']} 🎯\n"
            f"  TP2 hits:  {s['tp2']} 🚀\n"
            f"  SL hits:   {s['sl_hits']} ⛔\n\n"
            f"Last scan:   {last}\n"
            f"Pairs last:  {s['pairs_scanned']}\n"
            f"Tracking:    {len(self.s.active_trades)} trades",
            parse_mode=ParseMode.HTML,
        )

    async def trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        trades = self.s.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades.")
            return
        lines = [f"📡 <b>ACTIVE TRADES ({len(trades)})</b>\n"]
        for t in list(trades.values())[:15]:
            age = int((datetime.now() - t["timestamp"]).total_seconds() / 3600)
            tp1 = "✅" if t["tp1_hit"] else "⏳"
            tp2 = "✅" if t["tp2_hit"] else "⏳"
            lines.append(
                f"<b>{t['pair']}</b> {t['direction']}  "
                f"TP1{tp1} TP2{tp2}  Score:{t['score']}%  {age}h"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📚 <b>SMC SIGNAL BOT — STRATEGY</b>\n\n"
            "<b>Timeframes:</b>\n"
            "  • 4H → trend direction (HTF)\n"
            "  • 15M → entry precision (LTF)\n\n"
            "<b>Market Structure:</b>\n"
            "  • Break of Structure (BOS)\n"
            "  • Change of Character (CHOCH)\n"
            "  • Higher Highs / Higher Lows\n\n"
            "<b>Liquidity:</b>\n"
            "  • Sweep above swing highs\n"
            "  • Sweep below swing lows\n\n"
            "<b>Zones:</b>\n"
            "  • Bullish / Bearish Order Blocks\n"
            "  • Fair Value Gaps (FVG)\n\n"
            "<b>Confirmation:</b>\n"
            "  • RSI oversold/overbought\n"
            "  • Volume spike\n"
            "  • Engulfing / Pin bar\n\n"
            "<b>Score tiers:</b>\n"
            "  💎 PREMIUM ≥75%\n"
            "  🔥 HIGH    ≥60%\n"
            "  ✅ GOOD    ≥55%\n\n"
            "<b>Min R:R:</b> 1:2  |  Min Score: 55%",
            parse_mode=ParseMode.HTML,
        )


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

async def main():
    scanner = SMCScanner()
    cmds    = Commands(scanner)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmds.start))
    app.add_handler(CommandHandler("scan",   cmds.scan))
    app.add_handler(CommandHandler("stats",  cmds.stats))
    app.add_handler(CommandHandler("trades", cmds.trades))
    app.add_handler(CommandHandler("help",   cmds.help))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    logger.info("✅ Bot ready!")

    try:
        await scanner.run()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down…")
    finally:
        await scanner.close()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
