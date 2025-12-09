"""
Technical Analysis Engine - Calculates indicators and analyzes price patterns
"""

"""
# Quick test with fewer stocks (faster)
python stock_scanner.py --limit 5 --period 1mo

# Focus on tech stocks only
python stock_scanner.py --focus tech --limit 10

# Save results to JSON file
python stock_scanner.py --save
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD indicator"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Momentum"""
    return series - series.shift(period)

def calculate_price_change(df: pd.DataFrame, periods: list = [1, 5, 10, 20]) -> Dict[str, pd.Series]:
    """Calculate price changes over multiple periods"""
    changes = {}
    for period in periods:
        changes[f'change_{period}d'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    return changes

def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate volume-based indicators"""
    if 'volume' not in df.columns:
        return {}
    
    indicators = {}
    indicators['volume_sma'] = calculate_sma(df['volume'], 20)
    indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
    
    return indicators

def analyze_trend(df: pd.DataFrame) -> Dict:
    """Analyze trend direction and strength"""
    if len(df) < 50:
        return {'direction': 'unknown', 'strength': 0, 'ema_trend': 'unknown'}
    
    close = df['close']
    ema_9 = calculate_ema(close, 9)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    last_idx = len(df) - 1
    
    # Trend direction
    if ema_9.iloc[last_idx] > ema_21.iloc[last_idx] > ema_50.iloc[last_idx]:
        direction = 'up'
    elif ema_9.iloc[last_idx] < ema_21.iloc[last_idx] < ema_50.iloc[last_idx]:
        direction = 'down'
    else:
        direction = 'sideways'
    
    # Trend strength (0-100)
    price_change_20d = ((close.iloc[last_idx] - close.iloc[last_idx-20]) / close.iloc[last_idx-20]) * 100 if last_idx >= 20 else 0
    strength = min(100, max(0, abs(price_change_20d) * 2))
    
    return {
        'direction': direction,
        'strength': strength,
        'ema_trend': direction,
        'price_change_20d': price_change_20d
    }

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    if df.empty or len(df) < 20:
        return df
    
    result_df = df.copy()
    close = result_df['close']
    
    # RSI
    result_df['rsi'] = calculate_rsi(close, 14)
    
    # Moving Averages
    result_df['ema_9'] = calculate_ema(close, 9)
    result_df['ema_21'] = calculate_ema(close, 21)
    result_df['ema_50'] = calculate_ema(close, 50)
    result_df['sma_20'] = calculate_sma(close, 20)
    result_df['sma_50'] = calculate_sma(close, 50)
    
    # MACD
    macd_data = calculate_macd(close)
    result_df['macd'] = macd_data['macd']
    result_df['macd_signal'] = macd_data['signal']
    result_df['macd_histogram'] = macd_data['histogram']
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(close)
    result_df['bb_upper'] = bb_data['upper']
    result_df['bb_middle'] = bb_data['middle']
    result_df['bb_lower'] = bb_data['lower']
    
    # ATR
    result_df['atr'] = calculate_atr(result_df, 14)
    
    # Momentum
    result_df['momentum'] = calculate_momentum(close, 10)
    
    # Price changes
    price_changes = calculate_price_change(result_df)
    for key, value in price_changes.items():
        result_df[key] = value
    
    # Volume indicators
    volume_indicators = calculate_volume_indicators(result_df)
    for key, value in volume_indicators.items():
        result_df[key] = value
    
    return result_df

def get_current_signals(df: pd.DataFrame) -> Dict:
    """Get current trading signals from indicators"""
    if df.empty or len(df) < 20:
        return {}
    
    last_idx = len(df) - 1
    last = df.iloc[last_idx]
    
    signals = {
        'rsi': last.get('rsi', 50),
        'rsi_signal': 'oversold' if last.get('rsi', 50) < 30 else ('overbought' if last.get('rsi', 50) > 70 else 'neutral'),
        'ema_cross': 'bullish' if last.get('ema_9', 0) > last.get('ema_21', 0) else 'bearish',
        'macd_signal': 'bullish' if last.get('macd', 0) > last.get('macd_signal', 0) else 'bearish',
        'bb_position': 'upper' if last['close'] > last.get('bb_upper', last['close']) else ('lower' if last['close'] < last.get('bb_lower', last['close']) else 'middle'),
        'price_change_1d': last.get('change_1d', 0),
        'price_change_5d': last.get('change_5d', 0),
        'price_change_20d': last.get('change_20d', 0),
        'current_price': last['close'],
    }
    
    # Trend analysis
    trend = analyze_trend(df)
    signals.update(trend)
    
    return signals

if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="3mo")
    df.columns = [col.lower() for col in df.columns]
    
    df_with_indicators = calculate_all_indicators(df)
    signals = get_current_signals(df_with_indicators)
    
    print("Sample Technical Analysis:")
    print(f"RSI: {signals.get('rsi', 0):.2f}")
    print(f"Trend: {signals.get('direction', 'unknown')}")
    print(f"Price Change (20d): {signals.get('price_change_20d', 0):.2f}%")


