# Gold Scalping Documentation

This document explains the gold scalping system for trading XAUUSD (spot gold).

## 📁 Files

- **`gold_scalping.py`** - Interactive gold scalping analysis
- **`gold_scalping_live.py`** - Live monitoring with real-time signals
- **`gold_scalping_backtest.py`** - Backtesting system for historical analysis

## 🎯 Overview

The gold scalping system is designed for **short-term trading** of XAUUSD (spot gold) using:
- **Intraday intervals**: 5m, 15m, 30m for scalping
- **Strong zone identification**: Demand/supply zones with strength scoring
- **Pattern detection**: Chart patterns for entry/exit signals
- **Risk management**: Stop-loss and take-profit recommendations

## 🚀 Quick Start

### 1. Interactive Analysis

Run interactive gold scalping analysis:

```bash
python gold_scalping.py
```

**Features:**
- Prompts for interval selection (5m, 15m, 30m)
- Analyzes demand/supply zones with strength ratings
- Provides best buy/sell prices
- Calculates risk/reward ratios
- Shows scalping signals

**Output includes:**
- 🔥 Strong demand & supply zones with strength ratings
- 💰 Best buy price (optimal entry point)
- 💰 Best sell price (optimal exit point)
- 📊 Risk/reward ratio for scalping
- ⚡ Quick scalping signal (BUY/WAIT)

### 2. Live Monitoring

Monitor gold in real-time with automatic signal detection:

```bash
python gold_scalping_live.py
```

**Features:**
- Polls data every 60 seconds (configurable)
- Detects chart patterns automatically
- Prints BUY/SELL signals when conditions are met
- Manages position tracking (entry, stop-loss, take-profit)
- Shows exit signals (STOP, TARGET, TIME)

**Signal Logic:**
- **BUY Signal**: Chart pattern detected + RSI < 60 (confirmation)
- **SELL Signal**: Chart pattern detected + RSI > 40 (confirmation)
- **Exit Conditions**: Stop-loss hit, take-profit hit, or max hold time reached

**Pattern Detection:**
- Support/Resistance breakouts
- Trend structure (higher highs/lower lows)
- Consolidation breakouts
- Candlestick patterns (engulfing, hammer)
- Momentum shifts
- Price reversals

### 3. Backtesting

Test the strategy on historical data:

```bash
# Backtest 5m scalping for 60 days
python gold_scalping_backtest.py --period 60d --interval 5m

# Backtest daily for 5 years
python gold_scalping_backtest.py --period 5y --interval 1d

# Custom parameters
python gold_scalping_backtest.py --period 60d --interval 5m --max-hold-bars 12 --capital 10000
```

**Backtest Metrics:**
- Total return (%)
- Win rate (%)
- Average win/loss
- Risk/reward ratio
- Profit factor
- Max drawdown
- Pattern performance breakdown

## 📊 Data Sources

The system automatically tries data sources in this order:

1. **TradingView** (Recommended) - Most accurate, matches TradingView charts
   ```bash
   pip install pytradingview
   ```

2. **OANDA API** - Professional forex data
   ```bash
   # Set environment variable
   $env:OANDA_API_KEY="your_api_key_here"
   ```

3. **Yahoo Finance** (Fallback) - Uses GC=F (Gold Futures)
   - Already included via `yfinance`
   - Limited to ~60 days for 5m intervals

See `DATA_SOURCES.md` for detailed setup instructions.

## 🔍 Key Features

### Strong Zone Analysis

The system identifies **demand zones** (support) and **supply zones** (resistance) with strength scoring:

**Strength Factors:**
- Number of touches (max 40 points)
- Recent activity (max 30 points)
- Volume at zone (max 20 points)
- Bounce/rejection strength (max 10 points)

**Strength Levels:**
- 🔴 **VERY STRONG** (≥70): Excellent entry/exit point
- 🟠 **STRONG** (50-69): Good entry/exit point
- 🟡 **MODERATE** (30-49): Acceptable entry/exit point
- 🟢 **WEAK** (<30): Avoid or use with caution

### Best Buy/Sell Prices

The system automatically identifies optimal entry/exit points:

- **Best Buy Price**: Based on strongest demand zone below current price
- **Best Sell Price**: Based on strongest supply zone above current price
- **Entry Strategy**: 0.5% above demand zone (to confirm bounce)
- **Exit Strategy**: 0.5% below supply zone (to confirm rejection)

### Risk Management

**Stop-Loss:**
- Default: 3% below entry price
- Or: 3% below nearest demand zone

**Take-Profit:**
- Based on nearest supply zone
- Or: 1.5x risk (if no supply zone)

**Risk/Reward Ratio:**
- Excellent: ≥2:1
- Good: ≥1.5:1
- Low: <1.5:1 (consider tighter stop or higher target)

## 📈 Chart Pattern Detection

The live monitoring system detects multiple chart patterns:

### Buy Patterns

1. **SUPPORT_BREAKOUT** - Price broke above support level
2. **RESISTANCE_BREAKOUT** - Price broke above resistance (bullish continuation)
3. **UPTREND_STRUCTURE** - Higher highs and higher lows
4. **CONSOLIDATION_BREAKOUT_UP** - Broke out of tight range upward
5. **BULLISH_ENGULFING** - Bullish candlestick pattern
6. **HAMMER_REVERSAL** - Hammer pattern (reversal signal)
7. **SUPPORT_BOUNCE** - Bounce from support level
8. **MOMENTUM_UP** - 3+ consecutive higher closes
9. **REVERSAL_UP** - Reversal from recent low

### Sell Patterns

1. **RESISTANCE_REJECTION** - Price rejected from resistance
2. **DOWNTREND_STRUCTURE** - Lower highs and lower lows
3. **CONSOLIDATION_BREAKOUT_DOWN** - Broke down from tight range
4. **BEARISH_ENGULFING** - Bearish candlestick pattern
5. **MOMENTUM_DOWN** - 3+ consecutive lower closes
6. **REVERSAL_DOWN** - Rejection from recent high

**Pattern Strength:** 0-100 (higher = stronger signal)

## ⚙️ Configuration

### Interval Selection

**Recommended intervals for scalping:**
- **5m**: Very short-term scalping (5-minute candles)
- **15m**: Short-term scalping (15-minute candles)
- **30m**: Medium scalping (30-minute candles)

**Period recommendations:**
- **5m interval**: Use `5d` period (max ~60 days for Yahoo Finance)
- **15m interval**: Use `1mo` period
- **30m interval**: Use `1mo` period
- **1d interval**: Can use `5y` or longer

### Live Monitoring Settings

```python
monitor_gold_scalping(
    ticker="XAUUSD",
    interval="5m",
    period="5d",
    poll_seconds=60,      # Poll every 60 seconds
    max_hold_minutes=60   # Max hold time: 60 minutes
)
```

### Backtest Settings

```bash
python gold_scalping_backtest.py \
    --period 60d \
    --interval 5m \
    --max-hold-bars 12 \    # 12 bars = 60 min for 5m interval
    --capital 10000         # Starting capital
```

## 📊 Output Examples

### Interactive Analysis Output

```
🥇 GOLD SCALPING ANALYSIS - XAUUSD
======================================================================
Interval: 5m | Period: 5d
----------------------------------------------------------------------

🔥 STRONG DEMAND & SUPPLY ZONES
======================================================================
🟢 STRONG DEMAND ZONES (Best Buy Areas):
   1. $2650.50 - 🟠 STRONG
      Touches: 3 | Distance: 0.15% below current

🔴 STRONG SUPPLY ZONES (Best Sell Areas):
   1. $2665.75 - 🔴 VERY STRONG
      Touches: 5 | Distance: 0.20% above current

💰 BEST BUY & SELL PRICES FOR SCALPING
======================================================================
✅ BEST BUY PRICE: $2651.82
   Reason: Strong demand zone at $2650.50 (🟠 STRONG, 3 touches)
   💡 Wait for pullback: $0.18 (0.01%) lower

✅ BEST SELL PRICE: $2664.52
   Reason: Strong supply zone at $2665.75 (🔴 VERY STRONG, 5 touches)
   💰 Potential profit: $12.70 (0.48%)

📊 SCALPING RISK/REWARD:
   Entry: $2651.82
   Stop Loss: $2571.27 (Risk: $80.55)
   Take Profit: $2664.52 (Reward: $12.70)
   Risk/Reward Ratio: 1:0.16
   ⚠️  Low R:R ratio - consider tighter stop or higher target
```

### Live Monitoring Output

```
[2025-12-21 10:30:00] ⏱ Fetching latest data for XAUUSD...
[10:30:15] $2652.00 | RSI: 45.2 | Trend: UP | [BUY PATTERN: SUPPORT_BOUNCE]

================================================================================
[ANALYSIS] CHART PATTERN DETECTED
================================================================================
Price: $2652.00 | Ideal Entry: $2651.82
Target Exit: $2664.52 (Potential: $12.52)

[BUY PATTERN] SUPPORT_BOUNCE | Strength: 20/100
  • Bounce from support at $2650.50

[INDICATORS] RSI: 45.2 | Trend: UP
[CONFIRMATION] RSI PASSED (< 60) - BUY signal ready
================================================================================

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!  BUY SIGNAL TRIGGERED  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Pattern: SUPPORT_BOUNCE | Strength: 20/100
  • Bounce from support at $2650.50

ENTRY: $2652.00
STOP: $2571.27 | TARGET: $2664.52
RSI: 45.2 | Risk: $80.73 | Reward: $12.52
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## 🎯 Trading Strategy

### Entry Rules

1. **Wait for chart pattern** (buy or sell pattern detected)
2. **RSI confirmation**:
   - BUY: RSI < 60 (not extremely overbought)
   - SELL: RSI > 40 (not extremely oversold)
3. **Enter at best buy price** (or current price if pattern is strong)

### Exit Rules

1. **Take Profit**: Hit target (supply zone or 1.5R)
2. **Stop Loss**: Hit stop-loss (3% below entry or demand zone)
3. **Time Exit**: Max hold time reached (default: 60 minutes for 5m scalping)

### Position Sizing

- **Conservative**: 1% of capital per trade
- **Moderate**: 2% of capital per trade
- **Aggressive**: 3% of capital per trade (not recommended)

## ⚠️ Important Notes

1. **Not Financial Advice**: This system is for educational purposes only
2. **Data Limitations**: Yahoo Finance limits 5m data to ~60 days
3. **Market Hours**: Gold trades 24/5 (Monday-Friday)
4. **Slippage**: Real trading may have slippage (not accounted for in backtests)
5. **Commissions**: Factor in broker commissions for real trading
6. **Risk Management**: Always use stop-losses and proper position sizing

## 🔧 Troubleshooting

### "Could not fetch data for XAUUSD"

**Solutions:**
1. Install TradingView: `pip install pytradingview`
2. Set up OANDA API key: `$env:OANDA_API_KEY="your_key"`
3. Check internet connection
4. Try Yahoo Finance fallback (uses GC=F)

### "Not enough data"

**Solutions:**
- For 5m interval: Use period ≤ 60d
- For 1d interval: Can use longer periods (5y, 10y)
- Check data source availability

### "No strong zones found"

**Solutions:**
- Increase lookback period
- Try different interval (15m, 30m)
- Market may be in consolidation (wait for clearer structure)

## 📚 Related Documentation

- **`DATA_SOURCES.md`** - Detailed data source setup
- **`README.md`** - Main project documentation
- **`core/analysis/technical_analyzer.py`** - Technical indicator calculations
- **`core/data/forex_fetcher.py`** - Forex data fetching

## 💡 Tips for Best Results

1. **Use TradingView data** for most accurate results
2. **Focus on VERY STRONG zones** for highest probability trades
3. **Wait for pattern + RSI confirmation** before entering
4. **Use proper risk management** (stop-loss, position sizing)
5. **Backtest first** before live trading
6. **Monitor during active hours** (London/NY sessions for gold)

---

**Remember**: Gold scalping requires discipline and risk management. Always test strategies on historical data before live trading!

