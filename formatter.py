# ============================================================
#  formatter.py  –  Telegram message builder
# ============================================================

from strategy import TradeSignal


def _bar(score: int, length: int = 10) -> str:
    """Visual probability bar."""
    filled = round(score / 100 * length)
    return "█" * filled + "░" * (length - filled)


def _emoji_direction(direction: str) -> str:
    return "🟢" if direction == "LONG" else "🔴"


def _risk_emoji(risk: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")


def format_signal(sig: TradeSignal) -> str:
    emoji = _emoji_direction(sig.direction)
    risk_e = _risk_emoji(sig.risk_level)
    bar    = _bar(sig.score)

    lines = [
        f"{'='*38}",
        f"{emoji}  *{sig.pair}  —  {sig.direction} SIGNAL*",
        f"{'='*38}",
        "",
        f"💰 *ENTRY:*      `{sig.entry}`",
        f"🛑 *STOP LOSS:*  `{sig.stop_loss}`",
        f"🎯 *TP 1:*       `{sig.tp1}`",
        f"🚀 *TP 2:*       `{sig.tp2}`",
        f"⚖️  *R:R Ratio:*  `1 : {sig.rr_ratio}`",
        "",
        f"📊 *Probability:*  {sig.score}%  {bar}",
        f"{risk_e} *Risk Level:*   {sig.risk_level}",
        "",
        "─" * 36,
        "📋 *CONFIRMATIONS*",
        "─" * 36,
    ]

    for conf in sig.confirmations:
        lines.append(f"  • {conf}")

    lines += [
        "",
        "─" * 36,
        "⚠️  _This is not financial advice._",
        "_Always manage risk and use proper position sizing._",
        f"{'='*38}",
    ]

    return "\n".join(lines)


def format_scan_header(n_pairs: int, n_signals: int) -> str:
    return (
        f"🔍 *Scan complete* — {n_pairs} pairs checked, "
        f"*{n_signals} signal(s) found*"
    )


def format_no_signals() -> str:
    return (
        "😴 *No high-probability signals found.*\n"
        "_Markets need clearer setup — patience is a position._"
    )
