"""
Comprehensive Usage Examples

This file demonstrates all common usage patterns for the AI Stock Scanner.

Run from the ai-stock-scanner root directory:
    python examples/usage_examples.py [mode]

Modes:
    - signals: Get trading signals for a single stock
    - scan: Run full scanner (sequential)
    - parallel: Run full scanner (parallel)
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis import get_current_signals, calculate_all_indicators
from core.data import fetch_stock_data
from utils.format_signals import print_signals
from scanners import AgenticStockScanner


def example_get_signals():
    """Example 1: Get trading signals for a single stock"""
    print("=" * 70)
    print("EXAMPLE 1: Get Trading Signals for a Single Stock")
    print("=" * 70)
    
    ticker = "JEPQ"  # Change this to any ticker
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


def example_custom_analysis():
    """Example 4: Custom analysis script template"""
    print("=" * 70)
    print("EXAMPLE 4: Custom Analysis Template")
    print("=" * 70)
    
    # Your custom ticker list
    custom_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
    
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
    
    args = parser.parse_args()
    
    if args.mode == 'signals':
        example_get_signals()
    elif args.mode == 'scan':
        example_full_scan_sequential()
    elif args.mode == 'parallel':
        example_full_scan_parallel()
    elif args.mode == 'custom':
        example_custom_analysis()
    elif args.mode == 'all':
        example_get_signals()
        print("\n\n")
        example_custom_analysis()
    else:
        print("Available modes:")
        print("  signals  - Get trading signals for a single stock")
        print("  scan     - Run full scanner (sequential)")
        print("  parallel - Run full scanner (parallel)")
        print("  custom   - Custom analysis template")
        print("  all      - Run all examples")


if __name__ == "__main__":
    main()

