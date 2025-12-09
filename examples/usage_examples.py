"""
Comprehensive Usage Examples

This file demonstrates all common usage patterns for the AI Stock Scanner.

Run from the ai-stock-scanner root directory:
    python examples/usage_examples.py [mode] [--ticker TICKER] [--tickers TICKER1,TICKER2,...]

Modes:
    - signals: Get trading signals for a single stock (prompts for ticker)
    - scan: Run full scanner (sequential)
    - parallel: Run full scanner (parallel)
    - custom: Custom analysis for multiple tickers (prompts for tickers)
"""

import sys
import os
from typing import Optional, List
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis import get_current_signals, calculate_all_indicators
from core.data import fetch_stock_data
from utils.format_signals import print_signals
from scanners import AgenticStockScanner


def example_get_signals(ticker: Optional[str] = None):
    """Example 1: Get trading signals for a single stock"""
    print("=" * 70)
    print("EXAMPLE 1: Get Trading Signals for a Single Stock")
    print("=" * 70)
    
    # Get ticker from user input if not provided
    if ticker is None:
        ticker = input("\nEnter stock ticker symbol (e.g., AAPL, MSFT, JEPQ): ").strip().upper()
        if not ticker:
            print("⚠️ No ticker provided. Using default: AAPL")
            ticker = "AAPL"
    
    print(f"\nFetching data for {ticker}...")
    df = fetch_stock_data(ticker, period="3mo")
    
    if df is None or df.empty:
        print(f"Error: Could not fetch data for {ticker}")
        return
    
    print(f"✅ Fetched {len(df)} days of data")
    
    # Calculate indicators
    print("Calculating technical indicators...")
    df_with_indicators = calculate_all_indicators(df)
    print("✅ Indicators calculated")
    
    # Get signals
    print("Extracting trading signals...")
    signals = get_current_signals(df_with_indicators)
    print("✅ Signals extracted\n")
    
    # Print formatted output
    print_signals(signals, ticker)


def example_full_scan_sequential():
    """Example 2: Run full scanner sequentially"""
    print("=" * 70)
    print("EXAMPLE 2: Full Scanner (Sequential Mode)")
    print("=" * 70)
    print("\nThis scans multiple stocks one at a time.\n")
    
    # Initialize scanner (sequential by default)
    scanner = AgenticStockScanner(parallel=False)
    
    # Scan stocks
    results = scanner.scan_stocks(focus="all", period="3mo", parallel=False)
    
    # Print recommendations
    scanner.print_recommendations(limit=10)


def example_full_scan_parallel():
    """Example 3: Run full scanner with parallel processing"""
    print("=" * 70)
    print("EXAMPLE 3: Full Scanner (Parallel Mode)")
    print("=" * 70)
    print("\nThis analyzes multiple stocks simultaneously for faster results.\n")
    
    # Initialize scanner with parallel processing
    scanner = AgenticStockScanner(parallel=True, max_workers=5)
    
    # Scan stocks in parallel
    results = scanner.scan_stocks(focus="all", period="3mo", parallel=True)
    
    # Print recommendations
    scanner.print_recommendations(limit=10)


def example_custom_analysis(tickers: Optional[List[str]] = None):
    """Example 4: Custom analysis script template"""
    print("=" * 70)
    print("EXAMPLE 4: Custom Analysis Template")
    print("=" * 70)
    
    # Get tickers from user input if not provided
    if tickers is None:
        ticker_input = input("\nEnter stock ticker(s) separated by commas (e.g., AAPL,MSFT,NVDA): ").strip().upper()
        if ticker_input:
            custom_tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        else:
            print("⚠️ No tickers provided. Using defaults: AAPL, MSFT, NVDA, TSLA")
            custom_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
    else:
        custom_tickers = tickers
    
    print(f"\nAnalyzing custom tickers: {', '.join(custom_tickers)}\n")
    
    for ticker in custom_tickers:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}")
        print('='*60)
        
        # Fetch data
        df = fetch_stock_data(ticker, period="3mo")
        if df is None or df.empty:
            print(f"⚠️ Could not fetch {ticker}")
            continue
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(df)
        signals = get_current_signals(df_with_indicators)
        
        # Print formatted
        print_signals(signals, ticker)


def main():
    """Main function - run examples based on command line argument"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Stock Scanner Usage Examples')
    parser.add_argument('mode', nargs='?', 
                       choices=['signals', 'scan', 'parallel', 'custom', 'all'],
                       default='signals',
                       help='Example mode to run')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol (for signals mode)')
    parser.add_argument('--tickers', type=str, help='Comma-separated tickers (for custom mode, e.g., AAPL,MSFT,NVDA)')
    
    args = parser.parse_args()
    
    if args.mode == 'signals':
        example_get_signals(ticker=args.ticker)
    elif args.mode == 'scan':
        example_full_scan_sequential()
    elif args.mode == 'parallel':
        example_full_scan_parallel()
    elif args.mode == 'custom':
        tickers = None
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        example_custom_analysis(tickers=tickers)
    elif args.mode == 'all':
        example_get_signals(ticker=args.ticker)
        print("\n\n")
        tickers = None
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        example_custom_analysis(tickers=tickers)
    else:
        print("Available modes:")
        print("  signals  - Get trading signals for a single stock")
        print("  scan     - Run full scanner (sequential)")
        print("  parallel - Run full scanner (parallel)")
        print("  custom   - Custom analysis template")
        print("  all      - Run all examples")
        print("\nOptions:")
        print("  --ticker TICKER     - Specify ticker (for signals mode)")
        print("  --tickers T1,T2,... - Specify tickers (for custom mode)")


if __name__ == "__main__":
    main()

