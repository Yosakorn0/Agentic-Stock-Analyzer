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

def identify_demand_supply_zones(df: pd.DataFrame, lookback_period: int = 50, min_touches: int = 2) -> Dict:
    """
    Identify demand and supply zones
    
    Demand zones: Price levels with strong buying pressure (support)
    Supply zones: Price levels with strong selling pressure (resistance)
    
    Args:
        df: DataFrame with OHLCV data
        lookback_period: How many periods to look back
        min_touches: Minimum number of touches to qualify as a zone
    
    Returns:
        Dictionary with demand and supply zones
    """
    if df.empty or len(df) < lookback_period:
        return {
            'demand_zones': [],
            'supply_zones': [],
            'nearest_demand': None,
            'nearest_supply': None,
            'distance_to_demand': None,
            'distance_to_supply': None
        }
    
    # Use recent data
    recent_df = df.tail(lookback_period).copy()
    current_price = df['close'].iloc[-1]
    
    # Find swing highs (potential supply) and swing lows (potential demand)
    window = 5  # Look for local highs/lows within 5 periods
    
    demand_zones = []
    supply_zones = []
    
    # Identify swing lows (demand zones) - price bounced up from these levels
    for i in range(window, len(recent_df) - window):
        # Check if this is a local low
        is_low = True
        low_price = recent_df['low'].iloc[i]
        
        # Check surrounding periods
        for j in range(i - window, i + window + 1):
            if j != i and recent_df['low'].iloc[j] < low_price:
                is_low = False
                break
        
        if is_low:
            # Check if price bounced up after this low (demand zone)
            bounce_window = 3
            if i + bounce_window < len(recent_df):
                # Price should have increased after hitting this low
                if recent_df['close'].iloc[i + bounce_window] > low_price * 1.01:  # At least 1% bounce
                    demand_zones.append({
                        'price': float(low_price),
                        'strength': 'medium',
                        'touches': 1  # Simplified - would need more complex logic for multiple touches
                    })
    
    # Identify swing highs (supply zones) - price bounced down from these levels
    for i in range(window, len(recent_df) - window):
        # Check if this is a local high
        is_high = True
        high_price = recent_df['high'].iloc[i]
        
        # Check surrounding periods
        for j in range(i - window, i + window + 1):
            if j != i and recent_df['high'].iloc[j] > high_price:
                is_high = False
                break
        
        if is_high:
            # Check if price bounced down after this high (supply zone)
            bounce_window = 3
            if i + bounce_window < len(recent_df):
                # Price should have decreased after hitting this high
                if recent_df['close'].iloc[i + bounce_window] < high_price * 0.99:  # At least 1% drop
                    supply_zones.append({
                        'price': float(high_price),
                        'strength': 'medium',
                        'touches': 1
                    })
    
    # Remove duplicates and sort
    demand_zones = sorted(set([z['price'] for z in demand_zones]), reverse=True)  # Highest first
    supply_zones = sorted(set([z['price'] for z in supply_zones]), reverse=False)  # Lowest first
    
    # Find nearest zones to current price
    nearest_demand = None
    nearest_supply = None
    distance_to_demand = None
    distance_to_supply = None
    
    # Find nearest demand zone below current price
    for zone in demand_zones:
        if zone < current_price:
            nearest_demand = zone
            distance_to_demand = ((current_price - zone) / current_price) * 100
            break
    
    # Find nearest supply zone above current price
    for zone in supply_zones:
        if zone > current_price:
            nearest_supply = zone
            distance_to_supply = ((zone - current_price) / current_price) * 100
            break
    
    # Convert to list of dicts with metadata
    demand_zones_list = [{'price': p, 'type': 'demand', 'distance_pct': ((current_price - p) / current_price * 100) if p < current_price else ((p - current_price) / current_price * 100)} for p in demand_zones[:5]]  # Top 5
    supply_zones_list = [{'price': p, 'type': 'supply', 'distance_pct': ((p - current_price) / current_price * 100) if p > current_price else ((current_price - p) / current_price * 100)} for p in supply_zones[:5]]  # Top 5
    
    return {
        'demand_zones': demand_zones_list,
        'supply_zones': supply_zones_list,
        'nearest_demand': nearest_demand,
        'nearest_supply': nearest_supply,
        'distance_to_demand_pct': distance_to_demand,
        'distance_to_supply_pct': distance_to_supply
    }

def generate_buy_recommendation(df: pd.DataFrame, signals: Dict) -> Dict:
    """
    Generate buy recommendation with suggested entry price based on technical analysis
    
    Args:
        df: DataFrame with price data
        signals: Dictionary with technical signals
    
    Returns:
        Dictionary with buy recommendation and entry price suggestions
    """
    if df.empty or len(df) < 20:
        return {
            'recommendation': 'WAIT',
            'reason': 'Insufficient data',
            'suggested_entry_price': None,
            'entry_price_reason': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward_ratio': None
        }
    
    current_price = signals.get('current_price', 0)
    rsi = signals.get('rsi', 50)
    trend_direction = signals.get('direction', 'unknown')
    trend_strength = signals.get('strength', 0)
    ema_cross = signals.get('ema_cross', 'neutral')
    macd_signal = signals.get('macd_signal', 'neutral')
    nearest_demand = signals.get('nearest_demand')
    nearest_supply = signals.get('nearest_supply')
    distance_to_demand = signals.get('distance_to_demand_pct', 100)
    
    # Calculate recommendation score
    score = 0
    reasons = []
    
    # RSI analysis
    if rsi < 30:  # Oversold - good buying opportunity
        score += 30
        reasons.append("Oversold conditions (RSI < 30)")
    elif rsi < 40:
        score += 15
        reasons.append("Near oversold (RSI < 40)")
    elif rsi > 70:  # Overbought - wait
        score -= 20
        reasons.append("Overbought conditions (RSI > 70)")
    
    # Trend analysis
    if trend_direction == 'up' and trend_strength > 50:
        score += 25
        reasons.append("Strong uptrend")
    elif trend_direction == 'up':
        score += 15
        reasons.append("Uptrend")
    elif trend_direction == 'down' and trend_strength > 50:
        score -= 25
        reasons.append("Strong downtrend")
    elif trend_direction == 'down':
        score -= 15
        reasons.append("Downtrend")
    
    # EMA and MACD
    if ema_cross == 'bullish':
        score += 10
        reasons.append("Bullish EMA crossover")
    if macd_signal == 'bullish':
        score += 10
        reasons.append("Bullish MACD signal")
    
    # Price relative to demand zones
    if nearest_demand and distance_to_demand < 5:  # Within 5% of demand zone
        score += 15
        reasons.append(f"Near demand zone (${nearest_demand:,.2f})")
    elif nearest_demand and distance_to_demand < 10:
        score += 10
        reasons.append(f"Approaching demand zone (${nearest_demand:,.2f})")
    
    # Generate recommendation
    if score >= 60:
        recommendation = 'BUY'
    elif score >= 40:
        recommendation = 'CONSIDER BUY'
    elif score >= 20:
        recommendation = 'WATCH'
    else:
        recommendation = 'WAIT'
    
    # Suggest entry price
    suggested_entry_price = None
    entry_price_reason = None
    stop_loss = None
    take_profit = None
    risk_reward_ratio = None  # Initialize to avoid UnboundLocalError
    
    if recommendation in ['BUY', 'CONSIDER BUY']:
        # Strategy 1: If near demand zone, buy at or slightly above demand zone
        if nearest_demand and distance_to_demand < 5:
            suggested_entry_price = nearest_demand * 1.01  # 1% above demand zone
            entry_price_reason = f"Enter near demand zone (${nearest_demand:,.2f})"
        # Strategy 2: If oversold, buy at current price or slightly below
        elif rsi < 35:
            suggested_entry_price = current_price * 0.995  # 0.5% below current
            entry_price_reason = "Enter on oversold bounce"
        # Strategy 3: If in uptrend, buy on pullback to support (EMA 21 or demand zone)
        elif trend_direction == 'up' and nearest_demand:
            suggested_entry_price = max(nearest_demand, current_price * 0.98)
            entry_price_reason = "Buy on pullback in uptrend"
        # Strategy 4: Default - current price with small discount
        else:
            suggested_entry_price = current_price * 0.99
            entry_price_reason = "Enter at slight discount to current price"
        
        # Calculate stop loss (below nearest demand or 3% below entry)
        if nearest_demand:
            stop_loss = nearest_demand * 0.97  # 3% below demand zone
        else:
            stop_loss = suggested_entry_price * 0.97  # 3% below entry
        
        # Calculate take profit (near supply zone or 5-10% above entry)
        if nearest_supply and nearest_supply > suggested_entry_price:
            profit_pct = ((nearest_supply - suggested_entry_price) / suggested_entry_price) * 100
            if profit_pct < 15:  # Only use if reasonable
                take_profit = nearest_supply * 0.99  # Just below supply
            else:
                take_profit = suggested_entry_price * 1.08  # 8% profit target
        else:
            take_profit = suggested_entry_price * 1.08  # 8% profit target
        
        # Calculate risk/reward ratio
        risk = suggested_entry_price - stop_loss
        reward = take_profit - suggested_entry_price
        if risk > 0:
            risk_reward_ratio = reward / risk
        else:
            risk_reward_ratio = None
    else:
        # For WAIT/WATCH, suggest waiting for better entry
        if nearest_demand:
            suggested_entry_price = nearest_demand * 1.02
            entry_price_reason = f"Wait for pullback to demand zone (${nearest_demand:,.2f})"
        else:
            suggested_entry_price = current_price * 0.95
            entry_price_reason = "Wait for 5% pullback"
    
    return {
        'recommendation': recommendation,
        'recommendation_score': score,
        'reason': '; '.join(reasons) if reasons else 'Neutral conditions',
        'suggested_entry_price': float(suggested_entry_price) if suggested_entry_price else None,
        'entry_price_reason': entry_price_reason,
        'stop_loss': float(stop_loss) if stop_loss else None,
        'take_profit': float(take_profit) if take_profit else None,
        'risk_reward_ratio': float(risk_reward_ratio) if risk_reward_ratio else None
    }

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
    
    # Demand and supply zones
    zones = identify_demand_supply_zones(df)
    signals.update(zones)
    
    # Buy recommendation
    recommendation = generate_buy_recommendation(df, signals)
    signals.update(recommendation)
    
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


