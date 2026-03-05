# ============================================================
#  binance_client.py  –  Public Binance REST client (no key)
# ============================================================

import time
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com"   # Futures endpoint (public)

INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1d",
}


def get_klines(symbol: str, interval: str, limit: int = 200,
               retries: int = 3) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance Futures public endpoint.
    Returns a DataFrame with columns:
        open_time, open, high, low, close, volume, close_time
    """
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": INTERVAL_MAP.get(interval, interval),
        "limit": limit,
    }

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()

            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ])

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            return df

        except requests.exceptions.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s %s: %s",
                           attempt + 1, retries, symbol, interval, exc)
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch klines for {symbol} {interval}")


def get_ticker_price(symbol: str) -> float:
    """Return the current mark price from Binance Futures."""
    url = f"{BASE_URL}/fapi/v1/ticker/price"
    resp = requests.get(url, params={"symbol": symbol.upper()}, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["price"])


def get_all_futures_symbols() -> list[str]:
    """Return list of all active USDT-margined perpetual symbols."""
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return [
        s["symbol"] for s in data["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
    ]

def get_24h_tickers() -> dict:
    """
    Return {symbol: quoteVolume} for all USDT futures symbols.
    Uses the lightweight /fapi/v1/ticker/24hr endpoint.
    """
    url = f"{BASE_URL}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return {
        t["symbol"]: float(t["quoteVolume"])
        for t in resp.json()
        if t["symbol"].endswith("USDT")
    }


def get_liquid_symbols(min_volume_usdt: float = 5_000_000,
                       max_pairs=None) -> list:
    """
    Return active USDT perpetual symbols filtered by minimum 24h volume,
    sorted by volume descending (highest liquidity first).
    """
    active  = set(get_all_futures_symbols())
    volumes = get_24h_tickers()

    filtered = sorted(
        [s for s in active if volumes.get(s, 0) >= min_volume_usdt],
        key=lambda s: volumes.get(s, 0),
        reverse=True,
    )

    if max_pairs:
        filtered = filtered[:max_pairs]

    logger.info("Liquid symbols: %d / %d pass volume filter ($%s min).",
                len(filtered), len(active), f"{min_volume_usdt:,.0f}")
    return filtered
