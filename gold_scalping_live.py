"""
Live-ish gold scalping monitor.

This script DOES NOT place real trades. It:
- Periodically fetches recent GLD data (5m candles by default)
- Reuses the existing gold scalping logic (indicators + zones)
- Prints CLEAR terminal signals when:
    * A new BUY entry condition is met
    * An EXIT / SELL condition is met (target, stop, or max hold time)

Usage (from Agentic-Stock-Analyzer folder):
    python gold_scalping_live.py

Stop with Ctrl+C.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data.stock_fetcher import fetch_stock_data
from core.analysis.technical_analyzer import calculate_all_indicators, get_current_signals
from gold_scalping import find_best_buy_sell_prices


def monitor_gold_scalping(
    ticker: str = "XAUUSD=X",  # Yahoo spot gold vs USD; closest to Exness XAUUSD
    interval: str = "5m",
    period: str = "5d",
    poll_seconds: int = 60,
    max_hold_minutes: int = 60,
) -> None:
    """
    Monitor spot gold (XAUUSD) for scalping opportunities and print terminal signals.

    - Long-only, 5m scalping style
    - No broker integration, just prints BUY / EXIT messages
    - Uses same demand/supply zone logic as gold_scalping.py
    - Default: XAUUSD=X (Yahoo spot gold) – trade on Exness XAUUSD
    """

    print("=" * 80)
    print(f"🥇 GOLD SCALPING MONITOR - {ticker}")
    print("=" * 80)
    print(f"Interval: {interval} | History window: {period}")
    print(f"Polling every {poll_seconds} seconds | Max hold: {max_hold_minutes} minutes")
    print("Ctrl+C to stop.")
    print("=" * 80)

    bar_minutes = 5
    bars_per_trade = max(1, max_hold_minutes // bar_minutes)

    position_open = False
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_bar_index: Optional[int] = None

    last_signal_time: Optional[datetime] = None

    while True:
        try:
            now = datetime.now()
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] ⏱ Fetching latest data for {ticker}...")

            df = fetch_stock_data(ticker, period=period, interval=interval)
            if df is None or df.empty:
                print("❌ No data returned, skipping this cycle.")
                time.sleep(poll_seconds)
                continue

            df_ind = calculate_all_indicators(df.copy())
            closes = df_ind["close"].values
            highs = df_ind["high"].values
            lows = df_ind["low"].values
            index = df_ind.index

            i = len(df_ind) - 1  # use latest completed bar
            current_price = closes[i]
            current_time = index[i].to_pydatetime() if hasattr(index[i], "to_pydatetime") else index[i]

            window_df = df_ind.iloc[: i + 1]
            signals = get_current_signals(window_df)

            # Show brief status line
            print(
                f"Price: {current_price:.2f} | RSI: {signals.get('rsi', 0):.1f} "
                f"| Trend: {signals.get('direction', 'unknown')} "
                f"| Rec: {signals.get('recommendation', 'WAIT')}"
            )

            # Manage open position: check for exit signals
            if position_open:
                held_bars = (i - entry_bar_index) if entry_bar_index is not None else 0
                bar_high = highs[i]
                bar_low = lows[i]

                exit_reason = None
                exit_price = None

                if stop_loss is not None and bar_low <= stop_loss <= bar_high:
                    exit_reason = "STOP"
                    exit_price = stop_loss
                elif take_profit is not None and bar_low <= take_profit <= bar_high:
                    exit_reason = "TARGET"
                    exit_price = take_profit
                elif held_bars >= bars_per_trade:
                    exit_reason = "TIME"
                    exit_price = current_price

                if exit_price is not None:
                    pnl = exit_price - entry_price
                    print(
                        f"🚪 EXIT SIGNAL [{exit_reason}] "
                        f"Entry: {entry_price:.2f} at {entry_time}, "
                        f"Exit: {exit_price:.2f} at {current_time}, "
                        f"PnL: {pnl:.2f}"
                    )

                    position_open = False
                    entry_price = None
                    entry_time = None
                    stop_loss = None
                    take_profit = None
                    entry_bar_index = None

                    last_signal_time = now

                # When in a trade, skip new entries
                time.sleep(poll_seconds)
                continue

            # No open position: check for entry signal
            rec = signals.get("recommendation", "WAIT")
            rsi = signals.get("rsi", 50)
            trend = signals.get("direction", "unknown")

            # Always compute best prices so we can show diagnostics
            best_prices = find_best_buy_sell_prices(window_df, signals, current_price)
            best_buy = best_prices.get("best_buy_price") or current_price
            best_sell = best_prices.get("best_sell_price")

            # Basic diagnostic block so you can see *why* a signal did or didn't fire
            print("📊 Signal check (high-selectivity mode):")
            print(
                f"   Rec: {rec} (need BUY) | "
                f"RSI: {rsi:.1f} (need < 50) | "
                f"Trend: {trend} (avoid strong downtrend)"
            )
            print(
                f"   Best buy: {best_buy:.2f} | Current: {current_price:.2f}"
            )
            if best_sell is not None:
                print(f"   Best sell: {best_sell:.2f}")

            # Evaluate individual conditions
            # Allow a bit more distance to account for broker feed differences vs Yahoo (~1%)
            entry_threshold = 0.01  # 1.0% from ideal entry
            price_distance = abs(current_price - best_buy) / best_buy
            conditions = []

            # 1) Only take full BUY signals (skip CONSIDER BUY for higher win-rate)
            if rec == "BUY":
                conditions.append("✓ rec=BUY")
            else:
                conditions.append(f"✗ rec={rec} (need BUY)")

            # 2) Require RSI below 50 (tilt toward oversold / value entries)
            if rsi < 50:
                conditions.append("✓ rsi<50")
            else:
                conditions.append(f"✗ rsi={rsi:.1f}")

            # 3) Avoid clearly down-trending conditions for long-only scalps
            if trend != "down":
                conditions.append("✓ trend not 'down'")
            else:
                conditions.append("✗ trend=down")

            if price_distance <= entry_threshold:
                conditions.append(
                    f"✓ price distance {price_distance*100:.2f}% <= {entry_threshold*100:.1f}%"
                )
            else:
                conditions.append(
                    f"✗ price distance {price_distance*100:.2f}% > {entry_threshold*100:.1f}%"
                )

            print("   Conditions: " + " | ".join(conditions))

            # If all conditions are met, open a position
            if (
                rec == "BUY"
                and rsi < 50
                and trend != "down"
                and price_distance <= entry_threshold
            ):
                entry_price = float(current_price)
                entry_time = current_time
                entry_bar_index = i

                # Stop-loss from signals or 3% below entry
                stop_loss = float(signals.get("stop_loss", entry_price * 0.97))

                # Take-profit from best sell or 1.5R default
                if best_sell is not None:
                    take_profit = float(best_sell)
                else:
                    risk = entry_price - stop_loss
                    take_profit = float(entry_price + 1.5 * risk)

                print(
                    f"🟢 BUY SIGNAL ({rec}) "
                    f"Price: {entry_price:.2f} | "
                    f"Stop: {stop_loss:.2f} | Target: {take_profit:.2f} "
                    f"| RSI: {rsi:.1f}"
                )

                position_open = True
                last_signal_time = now
            else:
                print("⏸️  No BUY signal this bar; waiting for conditions.")

            time.sleep(poll_seconds)

        except KeyboardInterrupt:
            print("\n⚠️ Stopped by user (Ctrl+C).")
            break
        except Exception as e:
            print(f"\n❌ Error in monitor loop: {e}")
            # Small delay before retrying to avoid tight error loop
            time.sleep(poll_seconds)


if __name__ == "__main__":
    monitor_gold_scalping()


