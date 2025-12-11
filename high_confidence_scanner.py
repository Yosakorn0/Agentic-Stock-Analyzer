"""
High Confidence Buy Scanner

Scans through stocks and identifies those with high buy recommendation scores (60+).
Focuses on finding the best buy opportunities with strong signals.

Usage:
    python high_confidence_scanner.py
    python high_confidence_scanner.py --min-score 70
    python high_confidence_scanner.py --focus tech
    python high_confidence_scanner.py --tickers AAPL,MSFT,NVDA
"""

import sys
import os
from typing import List, Optional
import argparse
from datetime import datetime

# Import scanner
try:
    from stock_scanner import AgenticStockScanner
except ImportError:
    # Try modular version
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scanners.agentic_scanner import AgenticStockScanner


def scan_high_confidence_buys(
    min_score: int = 60,
    focus: str = "all",
    tickers: Optional[List[str]] = None,
    period: str = "3mo",
    interval: str = "1d"
):
    """
    Scan for stocks with high buy recommendation scores
    
    Args:
        min_score: Minimum score/confidence threshold (default: 60)
        focus: Focus area - "all", "tech", or "rising" (default: "all")
        tickers: Custom list of tickers to scan (optional)
        period: Time period for data (default: "3mo")
        interval: Data interval (default: "1d")
    
    Returns:
        List of high-confidence buy recommendations
    """
    print("=" * 80)
    print(f"🔍 HIGH CONFIDENCE BUY SCANNER")
    print("=" * 80)
    print(f"Minimum Score: {min_score}+")
    print(f"Focus: {focus.upper()}")
    print(f"Period: {period} | Interval: {interval}")
    print("=" * 80)
    print()
    
    # Initialize scanner
    scanner = AgenticStockScanner()
    
    # Scan stocks
    print("📊 Scanning stocks...")
    results = scanner.scan_stocks(
        tickers=tickers,
        period=period,
        interval=interval,
        focus=focus
    )
    
    if 'recommendations' not in results:
        print("❌ Error: No scan results available")
        return []
    
    # Filter for high-confidence buys
    high_confidence_buys = []
    all_recommendations = results.get('recommendations', [])
    
    for rec in all_recommendations:
        recommendation = rec.get('recommendation', 'WAIT')
        confidence = rec.get('confidence', 0)
        
        # Only include BUY or CONSIDER BUY with score >= min_score
        if recommendation in ['BUY', 'CONSIDER BUY'] and confidence >= min_score:
            high_confidence_buys.append(rec)
    
    return high_confidence_buys


def display_results(high_confidence_buys: List[dict], min_score: int):
    """
    Display high-confidence buy recommendations in a formatted way
    """
    if not high_confidence_buys:
        print("\n" + "=" * 80)
        print(f"❌ NO HIGH CONFIDENCE BUYS FOUND")
        print("=" * 80)
        print(f"No stocks found with score >= {min_score}")
        print("\n💡 Try:")
        print("   - Lowering the minimum score (--min-score 50)")
        print("   - Scanning different stocks (--focus rising)")
        print("   - Using a different time period")
        return
    
    # Sort by confidence (highest first)
    high_confidence_buys.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    print("\n" + "=" * 80)
    print(f"✅ FOUND {len(high_confidence_buys)} HIGH CONFIDENCE BUY OPPORTUNITIES")
    print("=" * 80)
    print()
    
    # Display summary table
    print(f"{'Rank':<6} {'Ticker':<8} {'Score':<8} {'Price':<12} {'Change 5d':<12} {'Risk':<10} {'Potential':<12}")
    print("-" * 80)
    
    for i, rec in enumerate(high_confidence_buys, 1):
        ticker = rec.get('ticker', 'N/A')
        confidence = rec.get('confidence', 0)
        price = rec.get('current_price', 0)
        change_5d = rec.get('price_change_5d', 0)
        risk = rec.get('risk_level', 'Medium')
        potential = rec.get('upside_potential', 'Medium')
        
        # Format values
        price_str = f"${price:,.2f}" if price else "N/A"
        change_str = f"{change_5d:+.2f}%" if change_5d else "N/A"
        
        # Choose icon based on score
        if confidence >= 80:
            icon = "🟢"
        elif confidence >= 70:
            icon = "🟡"
        else:
            icon = "🟠"
        
        print(f"{i:<6} {icon} {ticker:<6} {confidence:>3}%     {price_str:<12} {change_str:<12} {risk:<10} {potential:<12}")
    
    print("\n" + "=" * 80)
    print("📋 DETAILED RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Display detailed information for each
    for i, rec in enumerate(high_confidence_buys, 1):
        ticker = rec.get('ticker', 'N/A')
        name = rec.get('name', 'N/A')
        sector = rec.get('sector', 'Unknown')
        confidence = rec.get('confidence', 0)
        recommendation = rec.get('recommendation', 'WAIT')
        price = rec.get('current_price', 0)
        change_1d = rec.get('price_change_1d', 0)
        change_5d = rec.get('price_change_5d', 0)
        change_20d = rec.get('price_change_20d', 0)
        risk = rec.get('risk_level', 'Medium')
        potential = rec.get('upside_potential', 'Medium')
        reasoning = rec.get('reasoning', 'No reasoning provided')
        technical_score = rec.get('technical_score', 0)
        trend = rec.get('trend', 'unknown')
        rsi = rec.get('rsi', 0)
        
        # Choose icon
        if confidence >= 80:
            icon = "🟢"
            quality = "EXCELLENT"
        elif confidence >= 70:
            icon = "🟡"
            quality = "VERY GOOD"
        else:
            icon = "🟠"
            quality = "GOOD"
        
        print(f"{i}. {icon} {ticker} - {name}")
        print(f"   Sector: {sector}")
        print(f"   Score: {confidence}% ({quality}) | Recommendation: {recommendation}")
        print(f"   Current Price: ${price:,.2f}")
        print(f"   Price Changes: 1d: {change_1d:+.2f}% | 5d: {change_5d:+.2f}% | 20d: {change_20d:+.2f}%")
        print(f"   Risk Level: {risk} | Upside Potential: {potential}")
        print(f"   Technical Score: {technical_score:.1f} | Trend: {trend.upper()} | RSI: {rsi:.1f}")
        print(f"   Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"   Reasoning: {reasoning}")
        print()
    
    print("=" * 80)
    print("💡 TRADING TIPS")
    print("=" * 80)
    print("• Always use stop-losses (suggested: 2-3% below entry)")
    print("• Consider position sizing based on risk level:")
    print("  - Low Risk: 2-3% of capital")
    print("  - Medium Risk: 1-2% of capital")
    print("  - High Risk: 0.5-1% of capital")
    print("• Higher scores (80+) are stronger signals")
    print("• Monitor these stocks closely for entry opportunities")
    print("• Review technical indicators before entering")
    print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Scan for high-confidence buy opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all stocks with default 60+ score
  python high_confidence_scanner.py
  
  # Scan with higher threshold (70+)
  python high_confidence_scanner.py --min-score 70
  
  # Focus on tech stocks only
  python high_confidence_scanner.py --focus tech
  
  # Scan specific tickers
  python high_confidence_scanner.py --tickers AAPL,MSFT,NVDA,TSLA
  
  # Custom period and interval
  python high_confidence_scanner.py --period 1mo --interval 1d
        """
    )
    
    parser.add_argument(
        '--min-score',
        type=int,
        default=60,
        help='Minimum confidence score threshold (default: 60)'
    )
    
    parser.add_argument(
        '--focus',
        type=str,
        choices=['all', 'tech', 'rising'],
        default='all',
        help='Focus area: all, tech, or rising (default: all)'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated list of tickers to scan (e.g., AAPL,MSFT,NVDA)'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='3mo',
        help='Time period for data (default: 3mo)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        help='Data interval (default: 1d)'
    )
    
    args = parser.parse_args()
    
    # Parse tickers if provided
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    
    # Run scanner
    try:
        high_confidence_buys = scan_high_confidence_buys(
            min_score=args.min_score,
            focus=args.focus,
            tickers=tickers,
            period=args.period,
            interval=args.interval
        )
        
        # Display results
        display_results(high_confidence_buys, args.min_score)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Scan interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

