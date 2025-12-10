"""
Gold Scalping Example - Using AI Stock Scanner for Gold Trading

For scalping gold, you can use:
- GLD: Gold ETF (most reliable, works 24/7 data)
- GC=F: Gold Futures (may have market hours restrictions)

Recommended intervals for scalping:
- 5m: Very short-term scalping (5-minute candles)
- 15m: Short-term scalping (15-minute candles)
- 30m: Medium scalping (30-minute candles)

Run this script:
    python gold_scalping_example.py
"""

from core.analysis import get_current_signals, calculate_all_indicators
from core.data import fetch_stock_data
from utils.format_signals import print_signals
import sys
import pandas as pd
from typing import List, Dict

def calculate_zone_strength(df: pd.DataFrame, zone_price: float, zone_type: str, tolerance: float = 0.002) -> Dict:
    """
    Calculate strength of a demand/supply zone based on touches and volume
    
    Args:
        df: DataFrame with price data
        zone_price: Price level of the zone
        zone_type: 'demand' or 'supply'
        tolerance: Price tolerance (0.2% default) for counting touches
    
    Returns:
        Dictionary with strength metrics
    """
    touches = 0
    total_volume = 0
    recent_touches = 0
    bounce_strength = 0
    
    price_range = zone_price * (1 + tolerance) - zone_price * (1 - tolerance)
    lower_bound = zone_price * (1 - tolerance)
    upper_bound = zone_price * (1 + tolerance)
    
    for i in range(len(df)):
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        close = df['close'].iloc[i]
        volume = df.get('volume', pd.Series([1] * len(df))).iloc[i] if 'volume' in df.columns else 1
        
        # Check if price touched this zone
        touched = False
        if zone_type == 'demand':
            # Demand zone: price low touched the zone
            if lower_bound <= low <= upper_bound:
                touched = True
                # Measure bounce strength (how much price moved up after touching)
                if i < len(df) - 1:
                    next_close = df['close'].iloc[i + 1] if i + 1 < len(df) else close
                    bounce_pct = ((next_close - zone_price) / zone_price) * 100
                    bounce_strength += max(0, bounce_pct)
        else:  # supply
            # Supply zone: price high touched the zone
            if lower_bound <= high <= upper_bound:
                touched = True
                # Measure rejection strength (how much price moved down after touching)
                if i < len(df) - 1:
                    next_close = df['close'].iloc[i + 1] if i + 1 < len(df) else close
                    rejection_pct = ((zone_price - next_close) / zone_price) * 100
                    bounce_strength += max(0, rejection_pct)
        
        if touched:
            touches += 1
            total_volume += volume
            # Count recent touches (last 20% of data)
            if i >= len(df) * 0.8:
                recent_touches += 1
    
    # Calculate strength score (0-100)
    # Factors: number of touches, recent activity, volume, bounce/rejection strength
    touch_score = min(touches * 10, 40)  # Max 40 points for touches
    recent_score = min(recent_touches * 15, 30)  # Max 30 points for recent touches
    volume_score = min((total_volume / len(df)) / 1000000, 20) if len(df) > 0 else 0  # Max 20 points (scaled)
    bounce_score = min(bounce_strength / touches if touches > 0 else 0, 10)  # Max 10 points
    
    strength_score = touch_score + recent_score + volume_score + bounce_score
    
    # Categorize strength
    if strength_score >= 70:
        strength_level = "🔴 VERY STRONG"
    elif strength_score >= 50:
        strength_level = "🟠 STRONG"
    elif strength_score >= 30:
        strength_level = "🟡 MODERATE"
    else:
        strength_level = "🟢 WEAK"
    
    return {
        'strength_score': strength_score,
        'strength_level': strength_level,
        'touches': touches,
        'recent_touches': recent_touches,
        'avg_volume': total_volume / touches if touches > 0 else 0,
        'avg_bounce_strength': bounce_strength / touches if touches > 0 else 0
    }

def find_best_buy_sell_prices(df: pd.DataFrame, signals: Dict, current_price: float) -> Dict:
    """
    Find best buy and sell prices based on strong zones and current signals
    
    Returns:
        Dictionary with best buy/sell prices and reasons
    """
    demand_zones = signals.get('demand_zones', [])
    supply_zones = signals.get('supply_zones', [])
    
    # Analyze zone strengths
    strong_demand_zones = []
    strong_supply_zones = []
    
    # Find strong demand zones below current price (for buying)
    for zone in demand_zones:
        zone_price = zone.get('price', 0)
        if zone_price < current_price:
            strength = calculate_zone_strength(df, zone_price, 'demand')
            if strength['strength_score'] >= 40:  # At least moderate strength
                zone_info = zone.copy()
                zone_info.update(strength)
                strong_demand_zones.append(zone_info)
    
    # Find strong supply zones above current price (for selling)
    for zone in supply_zones:
        zone_price = zone.get('price', 0)
        if zone_price > current_price:
            strength = calculate_zone_strength(df, zone_price, 'supply')
            if strength['strength_score'] >= 40:  # At least moderate strength
                zone_info = zone.copy()
                zone_info.update(strength)
                strong_supply_zones.append(zone_info)
    
    # Sort by strength (strongest first) and proximity
    strong_demand_zones.sort(key=lambda x: (x['strength_score'], -x['distance_pct']), reverse=True)
    strong_supply_zones.sort(key=lambda x: (x['strength_score'], x['distance_pct']))
    
    # Best buy price: Strongest demand zone near current price
    best_buy_price = None
    best_buy_reason = None
    best_buy_zone = None
    
    if strong_demand_zones:
        best_buy_zone = strong_demand_zones[0]
        best_buy_price = best_buy_zone['price'] * 1.005  # 0.5% above zone for entry
        best_buy_reason = f"Strong demand zone at ${best_buy_zone['price']:.2f} ({best_buy_zone['strength_level']}, {best_buy_zone['touches']} touches)"
    elif signals.get('nearest_demand'):
        best_buy_price = signals['nearest_demand'] * 1.005
        best_buy_reason = f"Nearest demand zone at ${signals['nearest_demand']:.2f}"
    else:
        # Use recommended entry from signals
        best_buy_price = signals.get('suggested_entry_price', current_price * 0.995)
        best_buy_reason = "Based on technical analysis"
    
    # Best sell price: Strongest supply zone above current price
    best_sell_price = None
    best_sell_reason = None
    best_sell_zone = None
    
    if strong_supply_zones:
        best_sell_zone = strong_supply_zones[0]
        best_sell_price = best_sell_zone['price'] * 0.995  # 0.5% below zone for exit
        best_sell_reason = f"Strong supply zone at ${best_sell_zone['price']:.2f} ({best_sell_zone['strength_level']}, {best_sell_zone['touches']} touches)"
    elif signals.get('nearest_supply'):
        best_sell_price = signals['nearest_supply'] * 0.995
        best_sell_reason = f"Nearest supply zone at ${signals['nearest_supply']:.2f}"
    else:
        # Use take profit from signals
        best_sell_price = signals.get('take_profit', current_price * 1.05)
        best_sell_reason = "Based on technical analysis"
    
    return {
        'best_buy_price': best_buy_price,
        'best_buy_reason': best_buy_reason,
        'best_buy_zone': best_buy_zone,
        'best_sell_price': best_sell_price,
        'best_sell_reason': best_sell_reason,
        'best_sell_zone': best_sell_zone,
        'strong_demand_zones': strong_demand_zones[:3],  # Top 3
        'strong_supply_zones': strong_supply_zones[:3]   # Top 3
    }

def gold_scalping(ticker: str = "GLD", interval: str = "5m", period: str = "5d"):
    """
    Run scalping analysis for gold with strong zone identification
    
    Args:
        ticker: Gold ticker (GLD for ETF, GC=F for futures)
        interval: Data interval (5m, 15m, 30m for scalping)
        period: Time period (5d for 5m interval, 1mo for longer intervals)
    """
    print("=" * 70)
    print(f"🥇 GOLD SCALPING ANALYSIS - {ticker}")
    print("=" * 70)
    print(f"Interval: {interval} | Period: {period}")
    print("-" * 70)
    
    # Fetch intraday data for scalping
    print(f"\n📊 Fetching {interval} data for {ticker}...")
    df = fetch_stock_data(ticker, period=period, interval=interval)
    
    if df is None or df.empty:
        print(f"❌ Error: Could not fetch data for {ticker}")
        print("\n💡 TIPS:")
        print("   - Try GLD instead of GC=F")
        print("   - Check if market is open (for GC=F)")
        print("   - Try longer period (1mo) or different interval (15m)")
        return None
    
    print(f"✅ Fetched {len(df)} candles ({interval} interval)")
    print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
    print(f"💰 Latest price: ${df['close'].iloc[-1]:.2f}")
    
    # Calculate all technical indicators
    print("\n🔍 Calculating technical indicators...")
    df_with_indicators = calculate_all_indicators(df)
    
    # Get current trading signals
    print("📈 Analyzing signals...")
    signals = get_current_signals(df_with_indicators)
    
    current_price = df['close'].iloc[-1]
    
    # Find best buy/sell prices based on strong zones
    print("🎯 Analyzing zone strength...")
    best_prices = find_best_buy_sell_prices(df_with_indicators, signals, current_price)
    
    # Print formatted results
    print("\n" + "=" * 70)
    print_signals(signals, ticker)
    
    # Strong Demand & Supply Zones Section
    print("\n" + "=" * 70)
    print("🔥 STRONG DEMAND & SUPPLY ZONES")
    print("=" * 70)
    
    strong_demand = best_prices.get('strong_demand_zones', [])
    strong_supply = best_prices.get('strong_supply_zones', [])
    
    if strong_demand:
        print("\n🟢 STRONG DEMAND ZONES (Best Buy Areas):")
        for i, zone in enumerate(strong_demand, 1):
            zone_price = zone['price']
            strength = zone['strength_level']
            touches = zone['touches']
            distance = zone.get('distance_pct', 0)
            print(f"   {i}. ${zone_price:.2f} - {strength}")
            print(f"      Touches: {touches} | Distance: {abs(distance):.2f}% below current")
    else:
        print("\n⚠️  No strong demand zones found below current price")
        if signals.get('nearest_demand'):
            print(f"   Nearest demand: ${signals['nearest_demand']:.2f}")
    
    if strong_supply:
        print("\n🔴 STRONG SUPPLY ZONES (Best Sell Areas):")
        for i, zone in enumerate(strong_supply, 1):
            zone_price = zone['price']
            strength = zone['strength_level']
            touches = zone['touches']
            distance = zone.get('distance_pct', 0)
            print(f"   {i}. ${zone_price:.2f} - {strength}")
            print(f"      Touches: {touches} | Distance: {abs(distance):.2f}% above current")
    else:
        print("\n⚠️  No strong supply zones found above current price")
        if signals.get('nearest_supply'):
            print(f"   Nearest supply: ${signals['nearest_supply']:.2f}")
    
    # Best Buy/Sell Prices Section
    print("\n" + "=" * 70)
    print("💰 BEST BUY & SELL PRICES FOR SCALPING")
    print("=" * 70)
    
    best_buy = best_prices.get('best_buy_price')
    best_sell = best_prices.get('best_sell_price')
    
    if best_buy:
        print(f"\n✅ BEST BUY PRICE: ${best_buy:.2f}")
        print(f"   Reason: {best_prices.get('best_buy_reason', 'N/A')}")
        if best_buy < current_price:
            discount = ((current_price - best_buy) / current_price) * 100
            print(f"   💡 Wait for pullback: ${current_price - best_buy:.2f} ({discount:.2f}%) lower")
        elif best_buy > current_price:
            premium = ((best_buy - current_price) / current_price) * 100
            print(f"   ⚠️  Entry above current: ${best_buy - current_price:.2f} ({premium:.2f}%) higher")
        else:
            print(f"   ✅ Good entry at current price")
    else:
        print(f"\n⚠️  Could not determine best buy price")
        if signals.get('suggested_entry_price'):
            print(f"   Suggested entry: ${signals['suggested_entry_price']:.2f}")
    
    if best_sell:
        print(f"\n✅ BEST SELL PRICE: ${best_sell:.2f}")
        print(f"   Reason: {best_prices.get('best_sell_reason', 'N/A')}")
        if best_sell > current_price:
            profit = ((best_sell - current_price) / current_price) * 100
            print(f"   💰 Potential profit: ${best_sell - current_price:.2f} ({profit:.2f}%)")
        else:
            print(f"   ⚠️  Sell price below current - consider waiting")
    else:
        print(f"\n⚠️  Could not determine best sell price")
        if signals.get('take_profit'):
            print(f"   Take profit target: ${signals['take_profit']:.2f}")
    
    # Calculate risk/reward for scalping
    if best_buy and best_sell and best_buy < best_sell:
        stop_loss = signals.get('stop_loss', best_buy * 0.97)  # 3% below entry
        risk = best_buy - stop_loss
        reward = best_sell - best_buy
        if risk > 0:
            rr_ratio = reward / risk
            print(f"\n📊 SCALPING RISK/REWARD:")
            print(f"   Entry: ${best_buy:.2f}")
            print(f"   Stop Loss: ${stop_loss:.2f} (Risk: ${risk:.2f})")
            print(f"   Take Profit: ${best_sell:.2f} (Reward: ${reward:.2f})")
            print(f"   Risk/Reward Ratio: 1:{rr_ratio:.2f}")
            if rr_ratio >= 2:
                print(f"   ✅ Excellent R:R ratio for scalping!")
            elif rr_ratio >= 1.5:
                print(f"   ✅ Good R:R ratio")
            else:
                print(f"   ⚠️  Low R:R ratio - consider tighter stop or higher target")
    
    # Scalping-specific metrics
    print("\n" + "=" * 70)
    print("📊 SCALPING METRICS")
    print("=" * 70)
    print(f"RSI: {signals.get('rsi', 0):.2f} ({signals.get('rsi_signal', 'N/A').upper()})")
    if signals.get('rsi', 50) < 30:
        print("   ✅ Oversold - Good buy opportunity")
    elif signals.get('rsi', 50) > 70:
        print("   ⚠️  Overbought - Consider selling or waiting")
    
    print(f"\nTrend: {signals.get('direction', 'N/A').upper()} (Strength: {signals.get('strength', 0):.1f}/100)")
    print(f"Price Change (1 period): {signals.get('price_change_1d', 0) * 100:.2f}%")
    print(f"MACD Signal: {signals.get('macd_signal', 'N/A').upper()}")
    print(f"EMA Cross: {signals.get('ema_cross', 'N/A').upper()}")
    
    # Quick scalping recommendation
    print("\n" + "=" * 70)
    print("⚡ QUICK SCALPING SIGNAL")
    print("=" * 70)
    
    recommendation = signals.get('recommendation', 'WAIT')
    rsi = signals.get('rsi', 50)
    trend = signals.get('direction', 'neutral')
    
    if recommendation == 'BUY' and rsi < 40:
        print("🟢 STRONG BUY SIGNAL - Enter near demand zone")
        if best_buy and best_sell:
            print(f"   Buy: ${best_buy:.2f} | Target: ${best_sell:.2f}")
        else:
            print(f"   Buy near strong demand zone | Target: nearest supply zone")
    elif recommendation == 'BUY':
        print("🟡 BUY SIGNAL - Consider entering")
        if best_buy and best_sell:
            print(f"   Buy: ${best_buy:.2f} | Target: ${best_sell:.2f}")
        else:
            print(f"   Consider entering at current levels")
    elif recommendation == 'CONSIDER BUY' and rsi < 45:
        print("🟡 CONSIDER BUY - Watch for entry")
        if best_buy:
            print(f"   Ideal Buy: ${best_buy:.2f} if price pulls back")
        else:
            print(f"   Watch for pullback to demand zone")
    else:
        print("🔴 WAIT - Not ideal scalping conditions")
        if best_buy:
            print(f"   Wait for price to reach ${best_buy:.2f} (demand zone)")
        else:
            print("   Wait for better setup or clearer signals")
    
    return signals

if __name__ == "__main__":
    # Get user input
    print("🥇 Gold Scalping Setup")
    print("-" * 70)
    
    # Ticker selection
    ticker_choice = input("Enter gold ticker [GLD] (or GC=F): ").strip().upper()
    if not ticker_choice:
        ticker_choice = "GLD"
    
    # Interval selection
    print("\nSelect interval:")
    print("  1. 5m  (very short-term scalping)")
    print("  2. 15m (short-term scalping)")
    print("  3. 30m (medium scalping)")
    interval_choice = input("Enter choice [1-3] (default: 1): ").strip()
    
    interval_map = {"1": "5m", "2": "15m", "3": "30m", "": "5m"}
    interval = interval_map.get(interval_choice, "5m")
    
    # Period adjustment based on interval
    period_map = {"5m": "5d", "15m": "1mo", "30m": "1mo"}
    period = period_map.get(interval, "5d")
    
    # Run analysis
    try:
        gold_scalping(ticker=ticker_choice, interval=interval, period=period)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

