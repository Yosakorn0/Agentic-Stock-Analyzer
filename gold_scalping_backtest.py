"""
Gold Scalping Backtest - Test the gold scalping strategy on historical data

Usage:
    python gold_scalping_backtest.py --period 60d --interval 5m
    python gold_scalping_backtest.py --period 5y --interval 1d
    python gold_scalping_backtest.py --ticker XAUUSD --period 60d --interval 5m

Note: Yahoo Finance limits:
    - 5m interval: max ~60 days
    - 1d interval: up to 5y, 10y, or max
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Fix Windows terminal encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data.stock_fetcher import fetch_stock_data
from core.data.forex_fetcher import fetch_xauusd_data
from core.analysis.technical_analyzer import calculate_all_indicators, get_current_signals
from gold_scalping import find_best_buy_sell_prices
from gold_scalping_live import detect_chart_patterns


def backtest_gold_scalping(
    ticker: str = "XAUUSD",
    period: str = "60d",
    interval: str = "5m",
    max_hold_bars: int = 12,  # 12 bars = 60 minutes for 5m interval
    initial_capital: float = 10000.0,
) -> Dict:
    """
    Backtest the gold scalping strategy on historical data.
    
    Args:
        ticker: Gold ticker (XAUUSD only)
        period: Historical period (60d, 1mo, 3mo, 1y, 5y, etc.)
        interval: Data interval (5m, 15m, 1d, etc.)
        max_hold_bars: Maximum bars to hold a position
        initial_capital: Starting capital
        
    Returns:
        Dictionary with backtest results
    """
    # Always use XAUUSD as the display identifier
    display_ticker = "XAUUSD"
    
    print("=" * 80)
    print("🥇 GOLD SCALPING BACKTEST")
    print("=" * 80)
    print(f"Ticker: {display_ticker} | Period: {period} | Interval: {interval}")
    print(f"Max Hold: {max_hold_bars} bars | Initial Capital: ${initial_capital:,.2f}")
    print("=" * 80)
    
    # Try to fetch XAUUSD data from best available source
    # Priority: TradingView > OANDA > Yahoo Finance (GC=F)
    print("📊 Fetching XAUUSD data (TradingView > OANDA > Yahoo Finance)...", end=' ', flush=True)
    
    df = None
    data_source = None
    
    try:
        # Try the new forex fetcher first (TradingView/OANDA)
        df = fetch_xauusd_data(period=period, interval=interval, timeout=30)
        if df is not None and not df.empty:
            data_source = "TradingView/OANDA"
            print(f"✅ Success! (Source: {data_source})")
        else:
            # Fallback to Yahoo Finance
            print("⚠️  (TradingView/OANDA unavailable, trying Yahoo Finance...)", end=' ', flush=True)
            for fallback_ticker in ["GC=F", "GLD"]:
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(fetch_stock_data, fallback_ticker, period, interval)
                        df = future.result(timeout=30)
                    
                    if df is not None and not df.empty:
                        data_source = f"Yahoo Finance ({fallback_ticker})"
                        print(f"✅ Success! (Source: {data_source})")
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"❌ (failed: {str(e)[:50]})")
    
    if df is None or df.empty:
        print(f"\n❌ ERROR: Could not fetch data for {display_ticker}")
        print("\n💡 SETUP OPTIONS:")
        print("   1. TradingView (Recommended - matches TradingView charts):")
        print("      pip install pytradingview")
        print("   2. OANDA API (Free tier, professional data):")
        print("      - Sign up at https://www.oanda.com/us-en/trading/api/")
        print("      - Set environment variable: OANDA_API_KEY=your_key")
        print("   3. Yahoo Finance (Fallback - may have limitations):")
        print("      - Uses GC=F (Gold Futures) as data source")
        print("\n💡 TIPS:")
        print("   - For 5m interval, use period <= 60d")
        print("   - For 1d interval, you can use 5y or longer")
        return None
    print(f"✅ Fetched {len(df)} bars")
    print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("\n🔍 Calculating technical indicators...")
    df_ind = calculate_all_indicators(df.copy())
    
    if len(df_ind) < 50:
        print(f"❌ ERROR: Not enough data ({len(df_ind)} bars, need >= 50)")
        return None
    
    # Backtest simulation
    print("\n🔄 Running backtest simulation...")
    
    capital = initial_capital
    position = None  # {'entry_price': float, 'entry_bar': int, 'stop_loss': float, 'take_profit': float}
    trades = []
    equity_curve = []
    
    closes = df_ind['close'].values
    highs = df_ind['high'].values
    lows = df_ind['low'].values
    index = df_ind.index
    
    # Start from bar 20 (need enough history for indicators and patterns)
    start_bar = 20
    
    for i in range(start_bar, len(df_ind)):
        current_price = closes[i]
        current_high = highs[i]
        current_low = lows[i]
        current_time = index[i]
        
        # Get historical window up to current bar
        window_df = df_ind.iloc[:i+1]
        signals = get_current_signals(window_df)
        patterns = detect_chart_patterns(window_df, lookback=20)
        
        rsi = signals.get('rsi', 50)
        
        # Check exit conditions for open position
        if position is not None:
            held_bars = i - position['entry_bar']
            exit_reason = None
            exit_price = None
            
            # Check stop loss
            if current_low <= position['stop_loss'] <= current_high:
                exit_reason = "STOP"
                exit_price = position['stop_loss']
            # Check take profit
            elif current_low <= position['take_profit'] <= current_high:
                exit_reason = "TARGET"
                exit_price = position['take_profit']
            # Check max hold time
            elif held_bars >= max_hold_bars:
                exit_reason = "TIME"
                exit_price = current_price
            
            if exit_price is not None:
                # Close position
                pnl = exit_price - position['entry_price']
                pnl_pct = (pnl / position['entry_price']) * 100
                capital += pnl
                
                trades.append({
                    'entry_time': str(position['entry_time']),
                    'exit_time': str(current_time),
                    'entry_price': float(position['entry_price']),
                    'exit_price': float(exit_price),
                    'exit_reason': exit_reason,
                    'pnl': float(pnl),
                    'pnl_pct': float(pnl_pct),
                    'held_bars': int(held_bars),
                    'pattern': position.get('pattern', 'N/A')
                })
                
                position = None
        
        # Check entry conditions (only if no open position)
        if position is None:
            # BUY signal: Pattern + RSI confirmation
            buy_signal = False
            if patterns['buy_pattern'] and patterns['pattern_strength'] >= 20:
                if rsi < 60:  # RSI confirmation
                    buy_signal = True
            
            if buy_signal:
                # Calculate stop loss and take profit
                stop_loss_value = signals.get('stop_loss')
                stop_loss = float(stop_loss_value if stop_loss_value is not None else current_price * 0.97)
                best_prices = find_best_buy_sell_prices(window_df, signals, current_price)
                take_profit = best_prices.get('best_sell_price')
                if take_profit is None:
                    # Default 1.5R if no supply zone
                    risk = current_price - stop_loss
                    take_profit = float(current_price + 1.5 * risk)
                else:
                    take_profit = float(take_profit)
                
                # Enter position
                position = {
                    'entry_price': float(current_price),
                    'entry_bar': i,
                    'entry_time': current_time,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pattern': patterns['pattern_name']
                }
        
        # Record equity
        equity = capital
        if position is not None:
            # Add unrealized PnL
            unrealized_pnl = current_price - position['entry_price']
            equity += unrealized_pnl
        
        equity_curve.append({
            'time': str(current_time),
            'equity': float(equity),
            'capital': float(capital),
            'position_open': position is not None
        })
    
    # Close any remaining position at the end
    if position is not None:
        final_price = closes[-1]
        pnl = final_price - position['entry_price']
        pnl_pct = (pnl / position['entry_price']) * 100
        capital += pnl
        
        trades.append({
            'entry_time': str(position['entry_time']),
            'exit_time': str(index[-1]),
            'entry_price': float(position['entry_price']),
            'exit_price': float(final_price),
            'exit_reason': 'END_OF_DATA',
            'pnl': float(pnl),
            'pnl_pct': float(pnl_pct),
            'held_bars': int(len(df_ind) - position['entry_bar']),
            'pattern': position.get('pattern', 'N/A')
        })
    
    # Calculate metrics
    final_capital = capital
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = (len(winning_trades) / len(trades)) * 100
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        total_win = sum([t['pnl'] for t in winning_trades])
        total_loss = abs(sum([t['pnl'] for t in losing_trades]))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # Risk/Reward ratio
        if losing_trades:
            avg_risk = abs(avg_loss)
            avg_reward = avg_win
            rr_ratio = avg_reward / avg_risk if avg_risk > 0 else 0
        else:
            rr_ratio = 0
        
        # Max drawdown
        equity_values = [e['equity'] for e in equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_drawdown = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = ((peak - value) / peak) * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = t['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        rr_ratio = 0
        max_drawdown = 0
        exit_reasons = {}
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 BACKTEST RESULTS")
    print("=" * 80)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"\nTotal Trades: {len(trades)}")
    if trades:
        print(f"  ✅ Winning: {len(winning_trades)}")
        print(f"  ❌ Losing: {len(losing_trades)}")
        print(f"  📈 Win Rate: {win_rate:.1f}%")
        print(f"\nAverage Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Risk/Reward Ratio: 1:{rr_ratio:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        if exit_reasons:
            print(f"\nExit Reasons:")
            for reason, count in exit_reasons.items():
                print(f"  {reason}: {count}")
        
        # Pattern performance
        pattern_performance = {}
        for t in trades:
            pattern = t.get('pattern', 'N/A')
            if pattern not in pattern_performance:
                pattern_performance[pattern] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
            if t['pnl'] > 0:
                pattern_performance[pattern]['wins'] += 1
            else:
                pattern_performance[pattern]['losses'] += 1
            pattern_performance[pattern]['total_pnl'] += t['pnl']
        
        if pattern_performance:
            print(f"\nPattern Performance:")
            for pattern, perf in sorted(pattern_performance.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
                total = perf['wins'] + perf['losses']
                win_rate_pat = (perf['wins'] / total * 100) if total > 0 else 0
                print(f"  {pattern}: {perf['wins']}W/{perf['losses']}L ({win_rate_pat:.1f}%) | PnL: ${perf['total_pnl']:.2f}")
    else:
        print("⚠️  No trades executed during backtest period")
    
    print("=" * 80)
    
    # Return results dictionary (use display_ticker, not working_ticker)
    results = {
        'ticker': display_ticker,
        'data_source': working_ticker,  # Track which ticker was actually used for data
        'period': period,
        'interval': interval,
        'initial_capital': float(initial_capital),
        'final_capital': float(final_capital),
        'total_return': float(total_return),
        'total_trades': len(trades),
        'winning_trades': len(winning_trades) if trades else 0,
        'losing_trades': len(losing_trades) if trades else 0,
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'risk_reward_ratio': float(rr_ratio),
        'max_drawdown': float(max_drawdown),
        'trades': trades,
        'equity_curve': equity_curve,
        'exit_reasons': exit_reasons
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Backtest gold scalping strategy on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest 5m scalping for 60 days (max for intraday)
    python gold_scalping_backtest.py --period 60d --interval 5m
  
  # Backtest daily for 5 years
    python gold_scalping_backtest.py --period 5y --interval 1d
  
  # Use XAUUSD (default)
    python gold_scalping_backtest.py --ticker XAUUSD --period 60d --interval 5m

Note: Yahoo Finance limits:
  - 5m interval: max ~60 days
  - 1d interval: up to 5y, 10y, or max
        """
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        default='XAUUSD',
        help='Gold ticker (XAUUSD only). Default: XAUUSD'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='60d',
        help='Historical period (60d, 1mo, 3mo, 1y, 5y, etc.). Default: 60d'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='5m',
        help='Data interval (5m, 15m, 1d, etc.). Default: 5m'
    )
    
    parser.add_argument(
        '--max-hold-bars',
        type=int,
        default=12,
        help='Maximum bars to hold position (12 = 60 min for 5m). Default: 12'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital. Default: 10000'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save results to JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate period for intraday intervals
    if args.interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
        period_days = None
        if args.period.endswith('d'):
            period_days = int(args.period[:-1])
        elif args.period.endswith('mo'):
            period_days = int(args.period[:-2]) * 30
        elif args.period.endswith('y'):
            period_days = int(args.period[:-1]) * 365
        
        if period_days and period_days > 60:
            print(f"⚠️  WARNING: Intraday intervals (like {args.interval}) are limited to ~60 days by Yahoo Finance")
            print(f"   Requested period: {args.period} ({period_days} days)")
            print(f"   Will attempt to fetch, but may get less data than requested")
            print()
    
    try:
        results = backtest_gold_scalping(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            max_hold_bars=args.max_hold_bars,
            initial_capital=args.capital
        )
        
        if results and args.save:
            import json
            os.makedirs('results', exist_ok=True)
            filepath = args.save if args.save.endswith('.json') else f"{args.save}.json"
            if not os.path.dirname(filepath):
                filepath = os.path.join('results', filepath)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {filepath}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()