#!/usr/bin/env python3
# ============================================================
#  bot.py  –  Telegram Trading Signal Bot (entry point)
#             Supports scanning ALL USDT futures pairs with
#             async concurrency and rate limiting.
# ============================================================

import asyncio
import logging
import sys
import time

from telegram import Update, Bot
from telegram.ext import (
    Application, CommandHandler, ContextTypes, JobQueue
)
from telegram.constants import ParseMode

import config
from binance_client import get_liquid_symbols
from strategy       import analyse_pair
from formatter      import format_signal, format_scan_header, format_no_signals

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Prevent overlapping scans
_scan_lock = asyncio.Lock()


# ── Message helpers ───────────────────────────────────────────

def _split_message(text: str, limit: int = 4000) -> list:
    parts, current, length = [], [], 0
    for line in text.splitlines(keepends=True):
        if length + len(line) > limit:
            parts.append("".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line)
    if current:
        parts.append("".join(current))
    return parts


async def send_msg(bot: Bot, text: str, chat_id: str):
    try:
        for chunk in _split_message(text):
            await bot.send_message(
                chat_id=chat_id, text=chunk,
                parse_mode=ParseMode.MARKDOWN,
            )
            await asyncio.sleep(0.2)
    except Exception as exc:
        logger.error("send_message failed: %s", exc)


# ── Pair resolver ─────────────────────────────────────────────

def _resolve_pairs() -> list:
    if config.SCAN_MODE == "all":
        logger.info("Fetching all liquid USDT futures pairs …")
        try:
            pairs = get_liquid_symbols(
                min_volume_usdt=config.MIN_VOLUME_USDT,
                max_pairs=config.MAX_PAIRS,
            )
            logger.info("%d pairs queued.", len(pairs))
            return pairs
        except Exception as exc:
            logger.error("Symbol fetch failed: %s — using WATCHLIST", exc)
            return config.WATCHLIST
    return config.WATCHLIST


# ── Async concurrent scanner ──────────────────────────────────

async def _analyse_one(symbol: str, semaphore: asyncio.Semaphore,
                        delay: float):
    async with semaphore:
        await asyncio.sleep(delay)
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, analyse_pair, symbol)
        except Exception as exc:
            logger.warning("Error analysing %s: %s", symbol, exc)
            return None


async def _run_concurrent_scan(pairs: list) -> list:
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)
    tasks = [
        _analyse_one(sym, semaphore, i * config.REQUEST_DELAY)
        for i, sym in enumerate(pairs)
    ]
    results = await asyncio.gather(*tasks)
    signals = [r for r in results if r is not None]
    return sorted(signals, key=lambda s: s.score, reverse=True)


# ── Core scan routine ─────────────────────────────────────────

async def run_scan(bot: Bot, chat_id: str):
    if _scan_lock.locked():
        await send_msg(bot, "⏳ A scan is already in progress …", chat_id)
        return

    async with _scan_lock:
        t_start = time.monotonic()
        logger.info("Scan started (chat %s)", chat_id)

        await send_msg(
            bot,
            f"🔍 *Starting scan …*\n"
            f"Mode: `{config.SCAN_MODE.upper()}`  |  "
            f"Min 24h vol: `${config.MIN_VOLUME_USDT:,.0f}`",
            chat_id,
        )

        pairs = await asyncio.get_event_loop().run_in_executor(
            None, _resolve_pairs
        )
        if not pairs:
            await send_msg(bot, "❌ No pairs found.", chat_id)
            return

        await send_msg(
            bot,
            f"📊 Scanning *{len(pairs)} pairs* "
            f"({config.MAX_CONCURRENT} concurrent) …",
            chat_id,
        )

        signals = await _run_concurrent_scan(pairs)
        elapsed = round(time.monotonic() - t_start, 1)

        header = (
            f"✅ *Scan complete* in `{elapsed}s`\n"
            f"Pairs scanned: *{len(pairs)}*  |  Signals: *{len(signals)}*"
        )
        await send_msg(bot, header, chat_id)

        if signals:
            for sig in signals:
                await send_msg(bot, format_signal(sig), chat_id)
                await asyncio.sleep(0.4)
        else:
            await send_msg(bot, format_no_signals(), chat_id)

        logger.info("Scan done — %d signals in %.1fs.", len(signals), elapsed)


# ── Scheduled job ─────────────────────────────────────────────

async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    await run_scan(context.bot, config.TELEGRAM_CHAT_ID)


# ── Command handlers ──────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 *SMC Crypto Signal Bot*\n\n"
        "Commands:\n"
        "  /scan              — Run scan now\n"
        "  /status            — Show config\n"
        "  /setmode all       — Scan ALL USDT pairs\n"
        "  /setmode watchlist — Scan fixed watchlist only\n"
        "  /help              — This message\n\n"
        f"⏱ Auto-scan every *{config.SCAN_INTERVAL_MIN} min*\n"
        f"🌐 Current mode: *{config.SCAN_MODE.upper()}*"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await run_scan(context.bot, str(update.effective_chat.id))


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if config.SCAN_MODE == "all":
        pairs_info = (
            f"Mode: *ALL pairs*\n"
            f"  • Min 24h volume: `${config.MIN_VOLUME_USDT:,.0f}`\n"
            f"  • Max pairs cap: `{'none' if not config.MAX_PAIRS else config.MAX_PAIRS}`\n"
            f"  • Concurrency: `{config.MAX_CONCURRENT}` workers\n"
            f"  • Request stagger: `{config.REQUEST_DELAY}s`"
        )
    else:
        pairs_info = "Mode: *WATCHLIST*\n" + "\n".join(
            f"  • {p}" for p in config.WATCHLIST
        )

    text = (
        f"🤖 *Bot: ONLINE*\n\n"
        f"HTF: `{config.HTF}`  LTF: `{config.LTF}`\n"
        f"Scan every: `{config.SCAN_INTERVAL_MIN} min`\n"
        f"Min score: `{config.MIN_SCORE}%`  Min R:R: `1:{config.MIN_RR}`\n\n"
        f"📋 {pairs_info}"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_setmode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or args[0] not in ("all", "watchlist"):
        await update.message.reply_text(
            "Usage: `/setmode all` or `/setmode watchlist`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    config.SCAN_MODE = args[0]
    await update.message.reply_text(
        f"✅ Mode → *{config.SCAN_MODE.upper()}*",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


# ── Boot ──────────────────────────────────────────────────────

def main():
    logger.info("🚀 Starting SMC Signal Bot …")
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("setmode", cmd_setmode))
    app.add_handler(CommandHandler("help",    cmd_help))

    app.job_queue.run_repeating(
        scheduled_scan,
        interval=config.SCAN_INTERVAL_MIN * 60,
        first=15,
    )

    logger.info("✅ Bot polling … (Ctrl+C to stop)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
