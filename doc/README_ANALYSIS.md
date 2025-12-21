# Technical Analysis Documentation

## 📁 File

- **`core/analysis/technical_analyzer.py`** - Technical analysis engine

## 🎯 Purpose

The technical analyzer calculates comprehensive technical indicators and generates trading signals from price data. It's the foundation for all stock analysis in the system.

## 🚀 Quick Start

### Basic Usage

```python
from core.analysis import calculate_all_indicators, get_current_signals
from core.data import fetch_stock_data

# Fetch stock data
df = fetch_stock_data("AAPL", period="3mo")

# Calculate all indicators
df_with_indicators = calculate_all_indicators(df)

# Get current trading signals
signals = get_current_signals(df_with_indicators)

# Access signals
print(f"RSI: {signals['rsi']:.2f}")
print(f"Trend: {signals['direction']}")
print(f"Recommendation: {signals['recommendation']}")
```

## 📊 Technical Indicators

### 1. RSI (Relative Strength Index)

**Function**: `calculate_rsi(series, period=14)`

**Purpose**: Identifies overbought (>70) and oversold (<30) conditions

**Usage**:
```python
rsi = calculate_rsi(df['close'], period=14)
```

**Interpretation**:
- **RSI < 30**: Oversold (potential buy)
- **RSI > 70**: Overbought (potential sell)
- **30-70**: Neutral range

### 2. Moving Averages

#### EMA (Exponential Moving Average)

**Function**: `calculate_ema(series, period)`

**Usage**:
```python
ema_9 = calculate_ema(df['close'], 9)
ema_21 = calculate_ema(df['close'], 21)
ema_50 = calculate_ema(df['close'], 50)
```

**Common Periods**:
- **9 EMA**: Short-term trend
- **21 EMA**: Medium-term trend
- **50 EMA**: Long-term trend

#### SMA (Simple Moving Average)

**Function**: `calculate_sma(series, period)`

**Usage**:
```python
sma_20 = calculate_sma(df['close'], 20)
sma_50 = calculate_sma(df['close'], 50)
```

### 3. MACD (Moving Average Convergence Divergence)

**Function**: `calculate_macd(series, fast=12, slow=26, signal=9)`

**Returns**: Dictionary with `macd`, `signal`, and `histogram`

**Usage**:
```python
macd_data = calculate_macd(df['close'])
macd_line = macd_data['macd']
signal_line = macd_data['signal']
histogram = macd_data['histogram']
```

**Signals**:
- **Bullish**: MACD crosses above signal line
- **Bearish**: MACD crosses below signal line
- **Histogram**: Strength of momentum

### 4. Bollinger Bands

**Function**: `calculate_bollinger_bands(series, period=20, std_dev=2)`

**Returns**: Dictionary with `upper`, `middle`, and `lower` bands

**Usage**:
```python
bb = calculate_bollinger_bands(df['close'])
upper_band = bb['upper']
middle_band = bb['middle']  # SMA
lower_band = bb['lower']
```

**Signals**:
- **Price touches upper band**: Overbought
- **Price touches lower band**: Oversold
- **Band squeeze**: Low volatility (potential breakout)

### 5. ATR (Average True Range)

**Function**: `calculate_atr(df, period=14)`

**Purpose**: Measures volatility

**Usage**:
```python
atr = calculate_atr(df, period=14)
```

**Use Cases**:
- Stop-loss placement (2x ATR recommended)
- Position sizing based on volatility
- Volatility-based filters

### 6. Momentum

**Function**: `calculate_momentum(series, period=10)`

**Purpose**: Measures price momentum over N periods

**Usage**:
```python
momentum = calculate_momentum(df['close'], period=10)
```

### 7. Price Changes

**Function**: `calculate_price_change(df, periods=[1,5,10,20])`

**Returns**: Dictionary with change percentages

**Usage**:
```python
changes = calculate_price_change(df, periods=[1, 5, 10, 20])
change_1d = changes['change_1d']   # 1-day change %
change_5d = changes['change_5d']  # 5-day change %
change_20d = changes['change_20d']  # 20-day change %
```

### 8. Volume Indicators

**Function**: `calculate_volume_indicators(df)`

**Returns**: Dictionary with volume-based indicators

**Usage**:
```python
volume_indicators = calculate_volume_indicators(df)
volume_sma = volume_indicators['volume_sma']      # 20-period SMA of volume
volume_ratio = volume_indicators['volume_ratio']  # Current volume / SMA
```

**Signals**:
- **Volume ratio > 1.5**: Unusually high volume (potential breakout)
- **Volume ratio < 0.5**: Low volume (consolidation)

## 🔍 Analysis Functions

### 1. Trend Analysis

**Function**: `analyze_trend(df)`

**Returns**: Dictionary with trend direction and strength

**Usage**:
```python
trend = analyze_trend(df)
print(f"Direction: {trend['direction']}")  # 'up', 'down', or 'sideways'
print(f"Strength: {trend['strength']}")    # 0-100
```

**Trend Detection**:
- **Uptrend**: EMA 9 > EMA 21 > EMA 50
- **Downtrend**: EMA 9 < EMA 21 < EMA 50
- **Sideways**: Mixed EMA alignment

### 2. Demand & Supply Zones

**Function**: `identify_demand_supply_zones(df, lookback_period=50, min_touches=2)`

**Purpose**: Identifies key support (demand) and resistance (supply) levels

**Usage**:
```python
zones = identify_demand_supply_zones(df, lookback_period=50)

demand_zones = zones['demand_zones']      # List of demand zones
supply_zones = zones['supply_zones']      # List of supply zones
nearest_demand = zones['nearest_demand']  # Nearest demand below price
nearest_supply = zones['nearest_supply']  # Nearest supply above price
```

**Zone Structure**:
```python
{
    'price': 150.25,           # Zone price level
    'type': 'demand',          # 'demand' or 'supply'
    'distance_pct': 2.5       # Distance from current price (%)
}
```

### 3. Buy Recommendations

**Function**: `generate_buy_recommendation(df, signals)`

**Purpose**: Generates buy recommendation with entry price, stop-loss, and take-profit

**Returns**: Dictionary with:
- `recommendation`: 'BUY', 'CONSIDER BUY', 'WATCH', or 'WAIT'
- `score`: 0-100 (higher = better opportunity)
- `suggested_entry_price`: Optimal entry price
- `stop_loss`: Recommended stop-loss
- `take_profit`: Recommended take-profit
- `risk_reward_ratio`: Risk/reward ratio

**Usage**:
```python
signals = get_current_signals(df_with_indicators)
buy_rec = generate_buy_recommendation(df_with_indicators, signals)

if buy_rec['recommendation'] == 'BUY':
    print(f"Entry: ${buy_rec['suggested_entry_price']:.2f}")
    print(f"Stop: ${buy_rec['stop_loss']:.2f}")
    print(f"Target: ${buy_rec['take_profit']:.2f}")
    print(f"R:R Ratio: 1:{buy_rec['risk_reward_ratio']:.2f}")
```

## 📈 Current Signals

**Function**: `get_current_signals(df)`

**Purpose**: Extracts all current trading signals from the most recent data

**Returns**: Comprehensive dictionary with:

```python
{
    # RSI
    'rsi': 45.3,
    'rsi_signal': 'oversold',  # 'oversold', 'overbought', or 'neutral'
    
    # Trend
    'direction': 'up',          # 'up', 'down', or 'sideways'
    'strength': 75.5,          # 0-100
    'trend': 'up',
    
    # EMA
    'ema_cross': 'bullish',    # 'bullish', 'bearish', or 'neutral'
    
    # MACD
    'macd_signal': 'bullish',  # 'bullish', 'bearish', or 'neutral'
    
    # Bollinger Bands
    'bb_position': 'middle',   # 'upper', 'middle', or 'lower'
    
    # Price Changes
    'price_change_1d': 1.2,    # 1-day change %
    'price_change_5d': 3.5,   # 5-day change %
    'price_change_20d': 8.2,   # 20-day change %
    
    # Demand & Supply Zones
    'demand_zones': [...],     # List of demand zones
    'supply_zones': [...],      # List of supply zones
    'nearest_demand': 150.25,
    'nearest_supply': 155.50,
    'distance_to_demand_pct': 2.5,
    'distance_to_supply_pct': 1.8,
    
    # Buy Recommendation
    'recommendation': 'BUY',   # 'BUY', 'CONSIDER BUY', 'WATCH', 'WAIT'
    'suggested_entry_price': 152.50,
    'stop_loss': 148.00,
    'take_profit': 158.00,
    'risk_reward_ratio': 2.0,
    
    # Current Price
    'current_price': 152.75
}
```

## 🔄 Complete Workflow

```python
from core.data import fetch_stock_data
from core.analysis import calculate_all_indicators, get_current_signals
from utils.format_signals import print_signals

# 1. Fetch data
df = fetch_stock_data("AAPL", period="3mo")

# 2. Calculate indicators
df_with_indicators = calculate_all_indicators(df)

# 3. Get signals
signals = get_current_signals(df_with_indicators)

# 4. Print formatted output
print_signals(signals, "AAPL")

# 5. Access specific signals
if signals['recommendation'] == 'BUY':
    print(f"Entry: ${signals['suggested_entry_price']:.2f}")
    print(f"Stop: ${signals['stop_loss']:.2f}")
    print(f"Target: ${signals['take_profit']:.2f}")
```

## 📊 Indicator Combinations

### Bullish Setup

```python
signals = get_current_signals(df)

bullish = (
    signals['rsi'] < 50 and           # Not overbought
    signals['direction'] == 'up' and  # Uptrend
    signals['ema_cross'] == 'bullish' and  # Bullish EMA cross
    signals['macd_signal'] == 'bullish' and  # Bullish MACD
    signals['price_change_5d'] > 0    # Positive momentum
)

if bullish:
    print("Bullish setup detected!")
```

### Oversold Bounce

```python
signals = get_current_signals(df)

oversold_bounce = (
    signals['rsi'] < 30 and           # Oversold
    signals['nearest_demand'] and      # Near demand zone
    signals['price_change_1d'] < -2    # Recent drop
)

if oversold_bounce:
    print("Potential oversold bounce!")
```

### Breakout Setup

```python
signals = get_current_signals(df)

breakout = (
    signals['bb_position'] == 'upper' and  # Touching upper band
    signals.get('volume_ratio', 1) > 1.5 and  # High volume
    signals['price_change_5d'] > 5           # Strong momentum
)

if breakout:
    print("Potential breakout!")
```

## ⚙️ Configuration

### Custom Periods

```python
# Custom RSI period
rsi = calculate_rsi(df['close'], period=21)

# Custom MACD
macd = calculate_macd(df['close'], fast=8, slow=21, signal=5)

# Custom Bollinger Bands
bb = calculate_bollinger_bands(df['close'], period=30, std_dev=2.5)
```

### Zone Detection Settings

```python
# More sensitive zones (shorter lookback)
zones = identify_demand_supply_zones(df, lookback_period=30, min_touches=1)

# Less sensitive zones (longer lookback, more touches)
zones = identify_demand_supply_zones(df, lookback_period=100, min_touches=3)
```

## 📚 Related Files

- **`core/data/stock_fetcher.py`** - Data fetching
- **`core/analysis/ai_analyzer.py`** - AI analysis (uses technical signals)
- **`utils/format_signals.py`** - Signal formatting utilities
- **`stock_scanner.py`** - Main scanner (uses technical analyzer)

## 💡 Tips

1. **Use multiple indicators** - Don't rely on single indicator
2. **Check trend first** - Trend direction affects all signals
3. **Consider zones** - Demand/supply zones are key levels
4. **Volume confirmation** - High volume confirms breakouts
5. **Timeframe matters** - Different timeframes show different signals

---

**Note**: Technical analysis is not a guarantee of future performance. Always use proper risk management and combine with fundamental analysis when possible!

