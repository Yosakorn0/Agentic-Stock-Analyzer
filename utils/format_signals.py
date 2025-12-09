"""
Helper function to format trading signals for easy reading
"""

def format_signals(signals, ticker="STOCK"):
    """
    Format trading signals dictionary into a readable output.
    
    Args:
        signals: Dictionary from get_current_signals()
        ticker: Stock ticker symbol (optional)
    
    Returns:
        Formatted string
    """
    # Convert numpy types to regular Python types
    def to_float(value):
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value) if isinstance(value, (int, float)) else value
    
    output = []
    output.append("\n" + "=" * 70)
    output.append(f"  CURRENT TRADING SIGNALS FOR {ticker}")
    output.append("=" * 70)
    
    # Price and Basic Info
    current_price = to_float(signals.get('current_price', 0))
    output.append(f"\n📊 PRICE INFORMATION")
    output.append(f"   Current Price: ${current_price:,.2f}")
    
    # Technical Indicators
    rsi = to_float(signals.get('rsi', 50))
    rsi_signal = signals.get('rsi_signal', 'neutral')
    output.append(f"\n📈 TECHNICAL INDICATORS")
    output.append(f"   RSI: {rsi:.2f} ({rsi_signal.upper()})")
    output.append(f"   EMA Cross: {signals.get('ema_cross', 'neutral').upper()}")
    output.append(f"   MACD Signal: {signals.get('macd_signal', 'neutral').upper()}")
    output.append(f"   Bollinger Position: {signals.get('bb_position', 'middle').upper()}")
    
    # Trend Analysis
    direction = signals.get('direction', 'unknown')
    strength = to_float(signals.get('strength', 0))
    trend_emoji = "📈" if direction == 'up' else "📉" if direction == 'down' else "➡️"
    output.append(f"\n{trend_emoji} TREND ANALYSIS")
    output.append(f"   Direction: {direction.upper()}")
    output.append(f"   Strength: {strength:.1f}/100")
    
    # Price Changes
    output.append(f"\n💹 PRICE CHANGES")
    price_1d = to_float(signals.get('price_change_1d', 0))
    price_5d = to_float(signals.get('price_change_5d', 0))
    price_20d = to_float(signals.get('price_change_20d', 0))
    
    output.append(f"   1 day:  {price_1d:+.2f}%")
    output.append(f"   5 days: {price_5d:+.2f}%")
    output.append(f"   20 days: {price_20d:+.2f}%")
    
    # Demand and Supply Zones
    demand_zones = signals.get('demand_zones', [])
    supply_zones = signals.get('supply_zones', [])
    nearest_demand = signals.get('nearest_demand')
    nearest_supply = signals.get('nearest_supply')
    dist_demand = signals.get('distance_to_demand_pct')
    dist_supply = signals.get('distance_to_supply_pct')
    
    if demand_zones or supply_zones or nearest_demand or nearest_supply:
        output.append(f"\n🎯 DEMAND & SUPPLY ZONES")
        
        if nearest_demand is not None:
            dist_str = f" ({dist_demand:.2f}% below)" if dist_demand else ""
            output.append(f"   Nearest Demand Zone: ${nearest_demand:,.2f}{dist_str}")
        
        if nearest_supply is not None:
            dist_str = f" ({dist_supply:.2f}% above)" if dist_supply else ""
            output.append(f"   Nearest Supply Zone: ${nearest_supply:,.2f}{dist_str}")
        
        if demand_zones:
            output.append(f"\n   Demand Zones (Support Levels):")
            for zone in demand_zones[:3]:  # Show top 3
                zone_price = to_float(zone.get('price', 0))
                dist = zone.get('distance_pct', 0)
                direction = "below" if zone_price < current_price else "above"
                output.append(f"     ${zone_price:,.2f} ({abs(dist):.2f}% {direction})")
        
        if supply_zones:
            output.append(f"\n   Supply Zones (Resistance Levels):")
            for zone in supply_zones[:3]:  # Show top 3
                zone_price = to_float(zone.get('price', 0))
                dist = zone.get('distance_pct', 0)
                direction = "above" if zone_price > current_price else "below"
                output.append(f"     ${zone_price:,.2f} ({abs(dist):.2f}% {direction})")
    
    # Buy Recommendation
    recommendation = signals.get('recommendation', 'WAIT')
    recommendation_score = signals.get('recommendation_score', 0)
    entry_price = signals.get('suggested_entry_price')
    entry_reason = signals.get('entry_price_reason')
    stop_loss = signals.get('stop_loss')
    take_profit = signals.get('take_profit')
    risk_reward = signals.get('risk_reward_ratio')
    
    if recommendation:
        output.append(f"\n💰 BUY RECOMMENDATION")
        
        # Recommendation badge
        if recommendation == 'BUY':
            badge = "🟢 BUY"
        elif recommendation == 'CONSIDER BUY':
            badge = "🟡 CONSIDER BUY"
        elif recommendation == 'WATCH':
            badge = "🟠 WATCH"
        else:
            badge = "🔴 WAIT"
        
        output.append(f"   {badge} (Score: {recommendation_score}/100)")
        output.append(f"   Reason: {signals.get('reason', 'See analysis')}")
        
        # Entry price and strategy
        if entry_price:
            output.append(f"\n   📍 SUGGESTED ENTRY:")
            output.append(f"      Price: ${entry_price:,.2f}")
            if entry_reason:
                output.append(f"      Strategy: {entry_reason}")
            
            # Risk management
            if stop_loss or take_profit:
                output.append(f"\n   🛡️ RISK MANAGEMENT:")
                if stop_loss:
                    stop_loss_pct = ((current_price - stop_loss) / current_price) * 100
                    output.append(f"      Stop Loss: ${stop_loss:,.2f} ({stop_loss_pct:.2f}% below current)")
                if take_profit:
                    take_profit_pct = ((take_profit - current_price) / current_price) * 100
                    output.append(f"      Take Profit: ${take_profit:,.2f} ({take_profit_pct:.2f}% above current)")
                if risk_reward:
                    output.append(f"      Risk/Reward Ratio: 1:{risk_reward:.2f}")
        
        # Current price vs entry
        if entry_price:
            price_diff = entry_price - current_price
            price_diff_pct = (price_diff / current_price) * 100
            if abs(price_diff_pct) > 0.1:  # Only show if significant difference
                if price_diff < 0:
                    output.append(f"\n   ⚠️ Note: Entry price is ${abs(price_diff):,.2f} ({abs(price_diff_pct):.2f}%) below current price")
                    output.append(f"      Consider waiting for pullback or enter at market")
                else:
                    output.append(f"\n   ✅ Entry price is ${price_diff:,.2f} ({price_diff_pct:.2f}%) above current")
                    output.append(f"      Good entry opportunity")
    
    output.append("\n" + "=" * 70)
    
    return "\n".join(output)


def print_signals(signals, ticker="STOCK"):
    """Print formatted trading signals directly."""
    print(format_signals(signals, ticker))

