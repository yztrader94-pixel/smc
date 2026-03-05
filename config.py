# ============================================================
#  config.py  –  Central configuration for the SMC Trading Bot
# ============================================================

# ── Telegram ────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "7732870721:AAEHG3QJdo31S9sA8xjJzf-cXj6Tn4mo2uo"   # @BotFather token
TELEGRAM_CHAT_ID   = "7500072234"              # use @userinfobot to get

# ── Scan mode ───────────────────────────────────────────────
# SCAN_MODE options:
#   "all"       → fetch ALL active USDT perpetual futures from Binance (~300+)
#   "watchlist" → only scan the WATCHLIST below
SCAN_MODE = "all"

# Fallback / manual watchlist (used when SCAN_MODE = "watchlist")
WATCHLIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT",
]

# ── Full-scan filters (applied when SCAN_MODE = "all") ───────
# Minimum 24h quote volume in USDT to skip illiquid micro-caps
MIN_VOLUME_USDT    = 5_000_000      # $5M daily volume minimum
# Maximum number of pairs to scan after filtering (set None for unlimited)
MAX_PAIRS          = None

# ── Concurrency ──────────────────────────────────────────────
MAX_CONCURRENT     = 10             # parallel pair analyses
REQUEST_DELAY      = 0.15           # seconds between Binance requests

# ── Timeframes ──────────────────────────────────────────────
HTF = "4h"    # higher timeframe – trend direction
LTF = "15m"   # lower  timeframe – entry precision

# ── Candle limits ───────────────────────────────────────────
HTF_LIMIT = 200
LTF_LIMIT = 200

# ── Strategy thresholds ─────────────────────────────────────
RSI_PERIOD          = 14
RSI_OVERSOLD        = 35
RSI_OVERBOUGHT      = 65

VOLUME_SPIKE_MULT   = 1.6    # volume must be X × average to confirm
VOLUME_MA_PERIOD    = 20

SWING_LOOKBACK      = 10     # candles each side to define a swing pivot
OB_LOOKBACK         = 30     # how far back to search for order blocks
FVG_MIN_GAP_PCT     = 0.08   # minimum gap size as % of price

MIN_RR              = 2.0    # minimum risk-to-reward
SL_BUFFER_PCT       = 0.002  # 0.2 % buffer beyond structure for SL

MIN_SCORE           = 55     # only send signals scoring above this

# ── Scheduler ───────────────────────────────────────────────
SCAN_INTERVAL_MIN   = 15     # minutes between full scans
