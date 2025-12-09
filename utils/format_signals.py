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
    
    output.append("\n" + "=" * 70)
    
    return "\n".join(output)


def print_signals(signals, ticker="STOCK"):
    """Print formatted trading signals directly."""
    print(format_signals(signals, ticker))

