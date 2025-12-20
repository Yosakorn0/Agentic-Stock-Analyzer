"""
Live-ish gold scalping monitor.

This script DOES NOT place real trades. It:
- Periodically fetches recent XAUUSD data (5m candles by default)
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd

# Fix Windows terminal encoding for emojis
_original_print = print  # Save original print function

def safe_print(*args, **kwargs):
    """Print with fallback for Windows encoding issues"""
    try:
        _original_print(*args, **kwargs)
    except (UnicodeEncodeError, UnicodeError):
        # Fallback: replace emojis with text
        text = ' '.join(str(arg) for arg in args)
        # Replace common emojis with text equivalents
        replacements = {
            '🔍': '[TEST]', '✅': '[OK]', '❌': '[ERROR]', '💡': '[TIP]',
            '🥇': '[GOLD]', '🟢': '[BUY]', '🔴': '[SELL]', '⏸️': '[WAIT]',
            '📊': '[ANALYSIS]', '💰': '[PRICE]', '🎯': '[TARGET]', '📈': '[CHART]',
            '⚠️': '[WARN]', '🚪': '[EXIT]'
        }
        for emoji, text_repl in replacements.items():
            text = text.replace(emoji, text_repl)
        _original_print(text, **kwargs)

# Use safe_print if on Windows, otherwise regular print
if sys.platform == 'win32':
    import builtins
    builtins.print = safe_print

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data.stock_fetcher import fetch_stock_data
from core.data.forex_fetcher import fetch_xauusd_data
from core.analysis.technical_analyzer import calculate_all_indicators, get_current_signals
from gold_scalping import find_best_buy_sell_prices


def detect_chart_patterns(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect chart patterns and structure for scalping signals.
    Primary signal generator based on price action and structure.
    
    Returns:
        Dictionary with pattern signals: 'buy_pattern', 'sell_pattern', 'pattern_strength'
    """
    if len(df) < lookback:
        return {'buy_pattern': False, 'sell_pattern': False, 'pattern_strength': 0, 'pattern_name': 'INSUFFICIENT_DATA'}
    
    recent = df.tail(lookback).copy()
    current_price = recent['close'].iloc[-1]
    current_high = recent['high'].iloc[-1]
    current_low = recent['low'].iloc[-1]
    
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    opens = recent['open'].values
    
    buy_pattern = False
    sell_pattern = False
    pattern_strength = 0
    pattern_name = "NONE"
    pattern_details = []
    
    # 1. SUPPORT/RESISTANCE BREAKOUTS
    # Find recent support (local lows) and resistance (local highs)
    window = 5
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(recent) - window):
        # Check for support (local low with bounces)
        if lows[i] == min(lows[i-window:i+window+1]):
            if i + 3 < len(recent) and closes[i+3] > lows[i] * 1.005:  # Price bounced up
                support_levels.append(lows[i])
        
        # Check for resistance (local high with rejections)
        if highs[i] == max(highs[i-window:i+window+1]):
            if i + 3 < len(recent) and closes[i+3] < highs[i] * 0.995:  # Price rejected down
                resistance_levels.append(highs[i])
    
    # Check for support breakout (bullish)
    if support_levels:
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        if nearest_support:
            # Price broke above support with volume/strength
            if current_low > nearest_support * 1.002:  # 0.2% above support
                buy_pattern = True
                pattern_strength += 30
                pattern_name = "SUPPORT_BREAKOUT"
                pattern_details.append(f"Broke above support at ${nearest_support:.2f}")
    
    # Check for resistance breakout (bullish continuation)
    if resistance_levels:
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        if nearest_resistance:
            # Price broke above resistance
            if current_high > nearest_resistance * 1.001:
                buy_pattern = True
                pattern_strength += 40
                pattern_name = "RESISTANCE_BREAKOUT"
                pattern_details.append(f"Broke above resistance at ${nearest_resistance:.2f}")
    
    # Check for resistance rejection (bearish)
    if resistance_levels:
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        if nearest_resistance:
            # Price rejected from resistance
            distance_to_resistance = (nearest_resistance - current_price) / current_price
            if distance_to_resistance < 0.005:  # Within 0.5% of resistance
                if current_high >= nearest_resistance * 0.998 and closes[-1] < opens[-1]:  # Rejected with bearish candle
                    sell_pattern = True
                    pattern_strength += 35
                    pattern_name = "RESISTANCE_REJECTION"
                    pattern_details.append(f"Rejected from resistance at ${nearest_resistance:.2f}")
    
    # 2. TREND STRUCTURE (Higher Highs / Lower Lows)
    # Identify swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(3, len(recent) - 3):
        if highs[i] == max(highs[i-3:i+4]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-3:i+4]):
            swing_lows.append((i, lows[i]))
    
    # Check for higher highs and higher lows (uptrend structure)
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        recent_highs = sorted(swing_highs[-2:], key=lambda x: x[1])
        recent_lows = sorted(swing_lows[-2:], key=lambda x: x[1])
        
        # Higher highs and higher lows = uptrend structure
        if recent_highs[-1][1] > recent_highs[-2][1] and recent_lows[-1][1] > recent_lows[-2][1]:
            buy_pattern = True
            pattern_strength += 25
            if pattern_name == "NONE":
                pattern_name = "UPTREND_STRUCTURE"
            pattern_details.append("Higher highs & higher lows (uptrend)")
    
    # Lower highs and lower lows = downtrend structure
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        recent_highs = sorted(swing_highs[-2:], key=lambda x: x[1])
        recent_lows = sorted(swing_lows[-2:], key=lambda x: x[1])
        
        if recent_highs[-1][1] < recent_highs[-2][1] and recent_lows[-1][1] < recent_lows[-2][1]:
            sell_pattern = True
            pattern_strength += 25
            if pattern_name == "NONE":
                pattern_name = "DOWNTREND_STRUCTURE"
            pattern_details.append("Lower highs & lower lows (downtrend)")
    
    # 3. CONSOLIDATION BREAKOUT
    # Check if price was in tight range and broke out
    if len(recent) >= 10:
        recent_range = recent.tail(10)
        range_high = recent_range['high'].max()
        range_low = recent_range['low'].min()
        range_size = (range_high - range_low) / range_low
        
        # Tight consolidation (< 0.5% range)
        if range_size < 0.005:
            # Breakout above consolidation
            if current_high > range_high * 1.001:
                buy_pattern = True
                pattern_strength += 30
                if pattern_name == "NONE":
                    pattern_name = "CONSOLIDATION_BREAKOUT_UP"
                pattern_details.append(f"Broke out of consolidation (${range_low:.2f}-${range_high:.2f})")
            
            # Breakout below consolidation
            if current_low < range_low * 0.999:
                sell_pattern = True
                pattern_strength += 30
                if pattern_name == "NONE":
                    pattern_name = "CONSOLIDATION_BREAKOUT_DOWN"
                pattern_details.append(f"Broke down from consolidation (${range_low:.2f}-${range_high:.2f})")
    
    # 4. CANDLESTICK PATTERNS (last 3 candles)
    if len(recent) >= 3:
        last_3 = recent.tail(3)
        
        # Bullish engulfing
        if len(last_3) >= 2:
            prev_candle = last_3.iloc[-2]
            curr_candle = last_3.iloc[-1]
            
            # Bullish engulfing: previous bearish, current bullish and engulfs
            if prev_candle['close'] < prev_candle['open'] and curr_candle['close'] > curr_candle['open']:
                if curr_candle['open'] < prev_candle['close'] and curr_candle['close'] > prev_candle['open']:
                    buy_pattern = True
                    pattern_strength += 20
                    if pattern_name == "NONE":
                        pattern_name = "BULLISH_ENGULFING"
                    pattern_details.append("Bullish engulfing pattern")
            
            # Bearish engulfing
            if prev_candle['close'] > prev_candle['open'] and curr_candle['close'] < curr_candle['open']:
                if curr_candle['open'] > prev_candle['close'] and curr_candle['close'] < prev_candle['open']:
                    sell_pattern = True
                    pattern_strength += 20
                    if pattern_name == "NONE":
                        pattern_name = "BEARISH_ENGULFING"
                    pattern_details.append("Bearish engulfing pattern")
        
        # Hammer pattern (reversal)
        if len(last_3) >= 1:
            curr = last_3.iloc[-1]
            body = abs(curr['close'] - curr['open'])
            lower_shadow = min(curr['open'], curr['close']) - curr['low']
            upper_shadow = curr['high'] - max(curr['open'], curr['close'])
            
            # Hammer: small body, long lower shadow, small upper shadow
            if body > 0 and lower_shadow > body * 2 and upper_shadow < body * 0.5:
                if curr['close'] > curr['open']:  # Bullish hammer
                    buy_pattern = True
                    pattern_strength += 25
                    if pattern_name == "NONE":
                        pattern_name = "HAMMER_REVERSAL"
                    pattern_details.append("Hammer reversal pattern (bullish)")
    
    # 5. PRICE ACTION: Bounce from support / Rejection from resistance
    if support_levels:
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        if nearest_support:
            distance = (current_price - nearest_support) / nearest_support
            # More sensitive: within 0.5% of support and showing bullish action
            if 0 < distance < 0.005:
                # Strong bounce - bullish candle or higher close
                if closes[-1] > opens[-1] or closes[-1] > closes[-2] or (closes[-1] > nearest_support * 1.001):
                    buy_pattern = True
                    pattern_strength += 20
                    if pattern_name == "NONE":
                        pattern_name = "SUPPORT_BOUNCE"
                    pattern_details.append(f"Bounce from support at ${nearest_support:.2f}")
    
    # 6. MOMENTUM SHIFTS: Price acceleration patterns
    if len(recent) >= 5:
        # Check for accelerating upward momentum (3+ consecutive higher closes)
        recent_closes = closes[-5:]
        if len(recent_closes) >= 3:
            if all(recent_closes[i] > recent_closes[i-1] for i in range(1, min(4, len(recent_closes)))):
                buy_pattern = True
                pattern_strength += 15
                if pattern_name == "NONE":
                    pattern_name = "MOMENTUM_UP"
                pattern_details.append("3+ consecutive higher closes (momentum building)")
            
            # Check for accelerating downward momentum
            if all(recent_closes[i] < recent_closes[i-1] for i in range(1, min(4, len(recent_closes)))):
                sell_pattern = True
                pattern_strength += 15
                if pattern_name == "NONE":
                    pattern_name = "MOMENTUM_DOWN"
                pattern_details.append("3+ consecutive lower closes (momentum breaking)")
    
    # 7. PRICE REVERSAL: Recent low/high with reversal
    if len(recent) >= 5:
        # Check if we just made a recent low and are bouncing
        recent_low_idx = np.argmin(lows[-5:])
        if recent_low_idx < len(lows) - 2:  # Low was at least 2 bars ago
            recent_low = lows[-5:][recent_low_idx]
            if current_price > recent_low * 1.003 and closes[-1] > opens[-1]:  # 0.3% above low with bullish candle
                buy_pattern = True
                pattern_strength += 18
                if pattern_name == "NONE":
                    pattern_name = "REVERSAL_UP"
                pattern_details.append(f"Reversal from recent low at ${recent_low:.2f}")
        
        # Check if we just made a recent high and are rejecting
        recent_high_idx = np.argmax(highs[-5:])
        if recent_high_idx < len(highs) - 2:  # High was at least 2 bars ago
            recent_high = highs[-5:][recent_high_idx]
            if current_price < recent_high * 0.997 and closes[-1] < opens[-1]:  # 0.3% below high with bearish candle
                sell_pattern = True
                pattern_strength += 18
                if pattern_name == "NONE":
                    pattern_name = "REVERSAL_DOWN"
                pattern_details.append(f"Rejection from recent high at ${recent_high:.2f}")
    
    return {
        'buy_pattern': buy_pattern,
        'sell_pattern': sell_pattern,
        'pattern_strength': pattern_strength,
        'pattern_name': pattern_name,
        'pattern_details': pattern_details
    }


def monitor_gold_scalping(
    ticker: str = "XAUUSD",  # Uses XAUUSD exclusively
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
    - Uses XAUUSD as display identifier (tries XAUUSD formats, falls back to GC=F for data)
    """

    # Always use XAUUSD as the display identifier
    display_ticker = "XAUUSD"
    
    # Try to fetch XAUUSD data from best available source
    print("[TEST] Testing XAUUSD data sources (TradingView > OANDA > Yahoo Finance)...", end=' ', flush=True)
    
    working_ticker = None
    data_source = None
    test_df = None
    
    try:
        # Try the new forex fetcher first (TradingView/OANDA)
        test_df = fetch_xauusd_data(period=period, interval=interval, timeout=30)
        if test_df is not None and not test_df.empty:
            data_source = "TradingView/OANDA"
            working_ticker = display_ticker  # Use XAUUSD as identifier
            print(f"[OK] (Source: {data_source})")
        else:
            # Fallback to Yahoo Finance
            print("[SKIP] (TradingView/OANDA unavailable, trying Yahoo Finance...)", end=' ', flush=True)
            for fallback_ticker in ["GC=F", "GLD"]:
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(fetch_stock_data, fallback_ticker, period, interval)
                        test_df = future.result(timeout=30)
                    
                    if test_df is not None and not test_df.empty:
                        data_source = f"Yahoo Finance ({fallback_ticker})"
                        working_ticker = fallback_ticker
                        print(f"[OK] (Source: {data_source})")
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"[SKIP] (failed: {str(e)[:50]})")
    
    if working_ticker is None or test_df is None or test_df.empty:
        print(f"\n[ERROR] Could not fetch data for {display_ticker}")
        print("[TIP] Setup options:")
        print("  1. TradingView: pip install pytradingview")
        print("  2. OANDA: Set OANDA_API_KEY environment variable")
        print("  3. See DATA_SOURCES.md for details")
        return

    print("=" * 80)
    if working_ticker != display_ticker:
        print(f"GOLD SCALPING MONITOR - {display_ticker} (data: {working_ticker})")
    else:
        print(f"GOLD SCALPING MONITOR - {display_ticker}")
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
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] ⏱ Fetching latest data for {display_ticker}...")

            # Fetch data using the best available source
            try:
                if working_ticker == display_ticker and data_source and ("TradingView" in data_source or "OANDA" in data_source):
                    # Use forex fetcher for TradingView/OANDA
                    df = fetch_xauusd_data(period=period, interval=interval, timeout=30)
                else:
                    # Use stock fetcher for Yahoo Finance fallback
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(fetch_stock_data, working_ticker, period, interval)
                        df = future.result(timeout=30)
            except FutureTimeoutError:
                print("❌ Data fetch timed out, skipping this cycle.")
                time.sleep(poll_seconds)
                continue
            except Exception as e:
                print(f"❌ Error fetching data: {str(e)[:100]}, skipping this cycle.")
                time.sleep(poll_seconds)
                continue
            
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
            
            # DETECT CHART PATTERNS (PRIMARY SIGNAL)
            patterns = detect_chart_patterns(window_df, lookback=20)

            # Show brief status line (compact mode)
            pattern_status = ""
            if patterns['buy_pattern']:
                pattern_status = f" | [BUY PATTERN: {patterns['pattern_name']}]"
            elif patterns['sell_pattern']:
                pattern_status = f" | [SELL PATTERN: {patterns['pattern_name']}]"
            
            # Compact status line - only show if pattern detected or significant change
            rsi = signals.get('rsi', 50)
            if patterns['buy_pattern'] or patterns['sell_pattern']:
                # Show full line when pattern detected
                print(
                    f"\n[{now.strftime('%H:%M:%S')}] Price: ${current_price:.2f} | RSI: {rsi:.1f} "
                    f"| Trend: {signals.get('direction', 'unknown').upper()}{pattern_status}"
                )
            else:
                # Minimal output when no pattern - just timestamp and price
                print(f"[{now.strftime('%H:%M:%S')}] ${current_price:.2f} | RSI: {rsi:.1f} | No pattern", end='\r')

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

            # Only show detailed analysis when pattern detected or signal triggered
            show_details = patterns['buy_pattern'] or patterns['sell_pattern']
            
            if show_details:
                print("\n" + "=" * 80)
                print("[ANALYSIS] CHART PATTERN DETECTED")
                print("=" * 80)
                print(f"Price: ${current_price:.2f} | Ideal Entry: ${best_buy:.2f}")
                if best_sell is not None:
                    potential_profit = best_sell - current_price
                    print(f"Target Exit: ${best_sell:.2f} (Potential: ${potential_profit:.2f})")
                
                # Show CHART PATTERNS (PRIMARY SIGNAL)
                if patterns['buy_pattern']:
                    print(f"\n[BUY PATTERN] {patterns['pattern_name']} | Strength: {patterns['pattern_strength']}/100")
                    for detail in patterns['pattern_details']:
                        print(f"  • {detail}")
                elif patterns['sell_pattern']:
                    print(f"\n[SELL PATTERN] {patterns['pattern_name']} | Strength: {patterns['pattern_strength']}/100")
                    for detail in patterns['pattern_details']:
                        print(f"  • {detail}")
                
                # Show indicators
                print(f"\n[INDICATORS] RSI: {rsi:.1f} | Trend: {trend.upper()}")
                
                # Show signal confirmation status
                if patterns['buy_pattern']:
                    if rsi < 60:
                        print(f"[CONFIRMATION] RSI PASSED (< 60) - BUY signal ready")
                    else:
                        print(f"[CONFIRMATION] RSI FAILED (need < 60) - waiting")
                elif patterns['sell_pattern']:
                    if rsi > 40:
                        print(f"[CONFIRMATION] RSI PASSED (> 40) - SELL signal ready")
                    else:
                        print(f"[CONFIRMATION] RSI FAILED (need > 40) - waiting")
                
                print("=" * 80)

            # SCALPING SIGNAL LOGIC: Chart Patterns (Primary) + RSI (Confirmation)
            
            # BUY Signal: Pattern-based with RSI confirmation
            buy_signal = False
            buy_reason = []
            
            if patterns['buy_pattern'] and patterns['pattern_strength'] >= 20:
                # Pattern detected - check RSI confirmation
                if rsi < 60:  # Not extremely overbought (allows entries in uptrends)
                    buy_signal = True
                    buy_reason.append(f"Pattern: {patterns['pattern_name']} (Strength: {patterns['pattern_strength']})")
                    buy_reason.append(f"RSI confirmation: {rsi:.1f} (< 60)")
                else:
                    buy_reason.append(f"Pattern detected but RSI too high: {rsi:.1f} (need < 60)")
            
            # SELL Signal: Pattern-based with RSI confirmation
            sell_signal = False
            sell_reason = []
            
            if patterns['sell_pattern'] and patterns['pattern_strength'] >= 20:
                # Pattern detected - check RSI confirmation
                if rsi > 40:  # Not extremely oversold (allows exits in downtrends)
                    sell_signal = True
                    sell_reason.append(f"Pattern: {patterns['pattern_name']} (Strength: {patterns['pattern_strength']})")
                    sell_reason.append(f"RSI confirmation: {rsi:.1f} (> 40)")
                else:
                    sell_reason.append(f"Pattern detected but RSI too low: {rsi:.1f} (need > 40)")
            
            # Execute BUY signal (Pattern-based scalping)
            if buy_signal:
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

                # PROMINENT BUY SIGNAL ALERT
                print("\n" + "!" * 80)
                print("!" * 80)
                print("!" * 20 + "  BUY SIGNAL TRIGGERED  " + "!" * 38)
                print("!" * 80)
                print("!" * 80)
                print(f"Pattern: {patterns['pattern_name']} | Strength: {patterns['pattern_strength']}/100")
                for detail in patterns['pattern_details']:
                    print(f"  • {detail}")
                print(f"\nENTRY: ${entry_price:.2f}")
                print(f"STOP: ${stop_loss:.2f} | TARGET: ${take_profit:.2f}")
                print(f"RSI: {rsi:.1f} | Risk: ${entry_price - stop_loss:.2f} | Reward: ${take_profit - entry_price:.2f}")
                print("!" * 80)
                print("!" * 80 + "\n")

                position_open = True
                last_signal_time = now
            # Execute SELL signal (Pattern-based scalping)
            elif sell_signal and not position_open:  # Only if no open position
                # PROMINENT SELL SIGNAL ALERT
                print("\n" + "!" * 80)
                print("!" * 80)
                print("!" * 20 + "  SELL SIGNAL TRIGGERED  " + "!" * 37)
                print("!" * 80)
                print("!" * 80)
                print(f"Pattern: {patterns['pattern_name']} | Strength: {patterns['pattern_strength']}/100")
                for detail in patterns['pattern_details']:
                    print(f"  • {detail}")
                print(f"\nPrice: ${current_price:.2f} | RSI: {rsi:.1f}")
                if best_sell is not None:
                    print(f"Target Supply: ${best_sell:.2f}")
                print("[ACTION] Consider: Short entry or exit long positions")
                print("!" * 80)
                print("!" * 80 + "\n")
                last_signal_time = now
            else:
                if not buy_signal and not sell_signal:
                    if patterns['buy_pattern'] or patterns['sell_pattern']:
                        print(f"\n[WAIT] Pattern detected but waiting for RSI confirmation:")
                        if patterns['buy_pattern']:
                            print(f"  • {buy_reason[-1] if buy_reason else 'RSI needs to be < 60'}")
                        if patterns['sell_pattern']:
                            print(f"  • {sell_reason[-1] if sell_reason else 'RSI needs to be > 40'}")
                    # Don't print anything if no pattern - already shown minimal status above

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


