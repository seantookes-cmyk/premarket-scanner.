# premarket-scanner."""
Nasdaq premarket + opening-range 5-minute breakout scanner
(Alpaca free tier + Telegram alerts + optional Claude catalyst lookup)

What it does (Mon-Fri, 4:00 AM ET until RUN_UNTIL, default 10:00 AM ET):
  1. PREMARKET: finds Nasdaq gappers (gap % vs prior close, price range, volume).
     Every 5 min, alerts when the last COMPLETED 5-min bar closes above the
     premarket high (or below the premarket low) on above-average volume.
  2. OPENING RANGE: after the first 5-min candle (9:30-9:35) completes, alerts when
     a later 5-min bar closes above that candle's high (or below its low).
  3. CATALYST: on the first alert for a symbol, asks Claude (with web search) for
     the news driver and appends it to the alert. Needs ANTHROPIC_API_KEY.

Setup:
  pip install alpaca-py requests
  export APCA_API_KEY_ID=...          # alpaca.markets (paper account is fine)
  export APCA_API_SECRET_KEY=...
  export ALPACA_PAPER=true            # use false if your keys are from a LIVE account
  export TELEGRAM_BOT_TOKEN=...       # optional (create with @BotFather)
  export TELEGRAM_CHAT_ID=...         # optional
  export ANTHROPIC_API_KEY=...        # optional, enables catalyst lookup
  python premarket_scanner.py                 # waits for 4:00 AM ET, runs the session
  python premarket_scanner.py --now --once    # one test scan right now

NOTE: Free Alpaca data is the IEX feed, a small slice of total volume, so the
volume thresholds below are scaled down. With a paid SIP feed set
FEED = DataFeed.SIP and raise the volume thresholds ~20-50x.
Educational tool only. Not investment advice.
"""

import os
import sys
import time
from datetime import datetime, timedelta, date, time as dtime
from zoneinfo import ZoneInfo

import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus
from alpaca.trading.requests import GetAssetsRequest, GetCalendarRequest

# ---------------- Settings ----------------
FEED = DataFeed.IEX            # free tier. Use DataFeed.SIP if you pay for it
MIN_GAP_PCT = 3.0              # abs gap vs prior close
MIN_PRICE, MAX_PRICE = 2.0, 300.0
MIN_PM_VOLUME = 15_000         # cumulative premarket shares (IEX-scaled)
BAR_VOL_MULT = 2.0             # breakout bar volume vs average of prior bars
MIN_BAR_VOLUME = 3_000         # absolute floor for the breakout bar (IEX-scaled)
MIN_PRIOR_BARS = 3             # premarket bars needed before alerting
MAX_CANDIDATES = 60            # cap symbols tracked
LONGS, SHORTS = True, True
OPENING_RANGE = True           # enable opening-range breakouts after 9:35
RUN_UNTIL = dtime(10, 0)       # stop time (ET). Use dtime(9, 35) for premarket only
CATALYST_LOOKUP = True         # needs ANTHROPIC_API_KEY
CLAUDE_MODEL = "claude-sonnet-5-5"
NY = ZoneInfo("America/New_York")
OPEN = dtime(9, 30)
# ------------------------------------------

KEY = os.environ["APCA_API_KEY_ID"]
SECRET = os.environ["APCA_API_SECRET_KEY"]
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

data = StockHistoricalDataClient(KEY, SECRET)
PAPER = os.getenv("ALPACA_PAPER", "true").lower() != "false"  # set ALPACA_PAPER=false for live-account keys
trading = TradingClient(KEY, SECRET, paper=PAPER)  # used for asset list + calendar only

alerted = set()      # (symbol, bar_timestamp, label)
catalysts = {}       # symbol -> catalyst text (one lookup per symbol per run)


def notify(msg: str) -> None:
    print(msg, flush=True)
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data={"chat_id": TG_CHAT, "text": msg},
                timeout=10,
            )
        except requests.RequestException as e:
            print(f"Telegram error: {e}")


def get_catalyst(sym: str) -> str | None:
    """Ask Claude (with web search) for the news driver. Cached per symbol."""
    if not (CATALYST_LOOKUP and ANTHROPIC_KEY):
        return None
    if sym in catalysts:
        return catalysts[sym]
    prompt = (
        f"In at most 2 short sentences, what news or catalyst is moving the Nasdaq-listed "
        f"stock {sym} today ({datetime.now(NY):%B %d, %Y})? If there is no clear catalyst, "
        f"reply exactly: No clear catalyst found. No investment advice."
    )
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 400,
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        r.raise_for_status()
        text = " ".join(b["text"] for b in r.json()["content"] if b.get("type") == "text").strip()
        catalysts[sym] = text[:350] or "No clear catalyst found."
    except Exception as e:
        print(f"catalyst lookup error for {sym}: {e}")
        catalysts[sym] = None
    return catalysts[sym]


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    try:
        return len(trading.get_calendar(GetCalendarRequest(start=d, end=d))) > 0
    except Exception:
        return True  # if the calendar call fails, assume open


def nasdaq_universe() -> list[str]:
    assets = trading.get_all_assets(
        GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY,
            exchange=AssetExchange.NASDAQ,
        )
    )
    return [a.symbol for a in assets if a.tradable and a.symbol.isalpha() and len(a.symbol) <= 5]


def find_gappers(symbols: list[str]) -> dict[str, float]:
    """Return {symbol: gap_pct} for symbols passing the gap + price filters."""
    out = {}
    for i in range(0, len(symbols), 500):
        chunk = symbols[i : i + 500]
        try:
            snaps = data.get_stock_snapshots(StockSnapshotRequest(symbol_or_symbols=chunk, feed=FEED))
        except Exception as e:
            print(f"snapshot error: {e}")
            continue
        for sym, s in snaps.items():
            try:
                last = s.latest_trade.price
                prev = s.previous_daily_bar.close
            except AttributeError:
                continue
            if not prev or not last or not (MIN_PRICE <= last <= MAX_PRICE):
                continue
            gap = (last - prev) / prev * 100
            if abs(gap) >= MIN_GAP_PCT:
                out[sym] = gap
    ranked = sorted(out.items(), key=lambda kv: abs(kv[1]), reverse=True)[:MAX_CANDIDATES]
    return dict(ranked)


def emit(sym, label, direction, last, level, avg_vol, gap) -> None:
    key = (sym, last.timestamp, label)
    if key in alerted:
        return
    alerted.add(key)
    t = last.timestamp.astimezone(NY).strftime("%H:%M")
    msg = (
        f"{label} {direction} {sym}\n"
        f"5m bar {t} ET closed {last.close:.2f} vs level {level:.2f}\n"
        f"Bar vol {int(last.volume):,} ({last.volume / max(avg_vol, 1):.1f}x avg) | Gap {gap:+.1f}%\n"
        f"Idea: stop near bar low/high {last.low if direction == 'UP' else last.high:.2f}, size small."
    )
    cat = get_catalyst(sym)
    if cat:
        msg += f"\nCatalyst: {cat}"
    notify(msg)


def scan(gappers: dict[str, float], session_start: datetime) -> None:
    if not gappers:
        return
    now = datetime.now(NY)
    bars = data.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=list(gappers),
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=session_start,
            end=now,
            feed=FEED,
        )
    ).data

    for sym, blist in bars.items():
        # keep only COMPLETED bars (bar timestamp = start of the 5-min window)
        done = [b for b in blist if b.timestamp.astimezone(NY) + timedelta(minutes=5) <= now]
        if len(done) < 2:
            continue
        last = done[-1]
        last_t = last.timestamp.astimezone(NY).time()
        gap = gappers[sym]

        # ---------- Opening-range mode (bars after the 9:30 candle) ----------
        if OPENING_RANGE and last_t >= dtime(9, 35):
            or_bar = next((b for b in done if b.timestamp.astimezone(NY).time() == OPEN), None)
            if or_bar is None:
                continue
            prior = [b for b in done[:-1] if b.timestamp.astimezone(NY).time() >= OPEN]
            avg_vol = sum(b.volume for b in prior) / len(prior)
            if last.volume < max(MIN_BAR_VOLUME, BAR_VOL_MULT * avg_vol):
                continue
            if LONGS and last.close > or_bar.high:
                emit(sym, "OPENING RANGE", "UP", last, or_bar.high, avg_vol, gap)
            elif SHORTS and last.close < or_bar.low:
                emit(sym, "OPENING RANGE", "DOWN", last, or_bar.low, avg_vol, gap)
            continue

        # ---------- Premarket mode ----------
        if len(done) < MIN_PRIOR_BARS + 1 or sum(b.volume for b in done) < MIN_PM_VOLUME:
            continue
        prior = done[:-1]
        pm_high = max(b.high for b in prior)
        pm_low = min(b.low for b in prior)
        avg_vol = sum(b.volume for b in prior) / len(prior)
        if last.volume < max(MIN_BAR_VOLUME, BAR_VOL_MULT * avg_vol):
            continue
        if LONGS and last.close > pm_high:
            emit(sym, "PREMARKET", "UP", last, pm_high, avg_vol, gap)
        elif SHORTS and last.close < pm_low:
            emit(sym, "PREMARKET", "DOWN", last, pm_low, avg_vol, gap)


def sleep_to_next_5min() -> None:
    now = datetime.now(NY)
    nxt = (now + timedelta(minutes=5 - (now.minute % 5))).replace(second=8, microsecond=0)
    time.sleep(max(1, (nxt - now).total_seconds()))


def main() -> None:
    force = "--now" in sys.argv
    once = "--once" in sys.argv
    today = datetime.now(NY).date()
    if not force and not is_trading_day(today):
        print("Market closed today. Exiting.")
        return

    start = datetime.combine(today, dtime(4, 0), NY)
    end = datetime.combine(today, RUN_UNTIL, NY)

    if not force:
        while datetime.now(NY) < start:
            time.sleep(30)

    notify(f"Scanner started {datetime.now(NY):%Y-%m-%d %H:%M} ET")
    universe = nasdaq_universe()
    print(f"Universe: {len(universe)} Nasdaq symbols")

    tracked: dict[str, float] = {}
    while force or datetime.now(NY) <= end:
        try:
            # refresh gappers until the open, then freeze the list for opening-range tracking
            if not tracked or datetime.now(NY).time() < OPEN:
                tracked = find_gappers(universe)
            print(f"{datetime.now(NY):%H:%M} tracking {len(tracked)} symbols", flush=True)
            scan(tracked, start)
        except Exception as e:
            print(f"cycle error: {e}")
        if once:
            break
        sleep_to_next_5min()

    notify("Scanner finished for today.")


if __name__ == "__main__":
    main()
