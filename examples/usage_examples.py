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
import time
from typing import Optional, List
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import tqdm for progress bars, fallback to simple progress if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple progress bar fallback
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", unit="", ncols=80, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else None)
            self.desc = desc
            self.unit = unit
            self.current = 0
            self.start_time = time.time()
            if self.iterable:
                self.iterable = iter(self.iterable)
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.iterable:
                try:
                    item = next(self.iterable)
                    self.current += 1
                    self._update()
                    return item
                except StopIteration:
                    self._finish()
                    raise
            else:
                raise StopIteration
        
        def update(self, n=1):
            self.current += n
            self._update()
        
        def _update(self):
            if self.total:
                percent = (self.current / self.total) * 100
                elapsed = time.time() - self.start_time
                if self.current > 0:
                    rate = self.current / elapsed
                    eta = (self.total - self.current) / rate if rate > 0 else 0
                    bar_length = 30
                    filled = int(bar_length * self.current / self.total)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({percent:.1f}%) "
                          f"| Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end='', flush=True)
        
        def _finish(self):
            elapsed = time.time() - self.start_time
            print(f"\r{self.desc}: [{'█' * 30}] {self.current}/{self.total} (100.0%) "
                  f"| Elapsed: {elapsed:.1f}s | Complete!{' ' * 20}")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self._finish()
    
    # Add write function for tqdm compatibility (tqdm.write is a module-level function)
    def tqdm_write(s, file=None, end="\n"):
        """Write function for tqdm compatibility"""
        print(s, end=end, file=file, flush=True)
    
    # Attach write to tqdm class as a static method
    tqdm.write = tqdm_write

from core.analysis import get_current_signals, calculate_all_indicators
from core.data import fetch_stock_data
from utils.format_signals import print_signals
from scanners import AgenticStockScanner


def example_get_signals(ticker: Optional[str] = None):
    """Example 1: Get trading signals for a single stock"""
    print("=" * 70)
    print("EXAMPLE 1: Get Trading Signals for a Single Stock")
    print("=" * 70)
    
    start_time = time.time()
    
    # Get ticker from user input if not provided
    if ticker is None:
        ticker = input("\nEnter stock ticker symbol (e.g., AAPL, MSFT, JEPQ): ").strip().upper()
        if not ticker:
            print("⚠️ No ticker provided. Using default: AAPL")
            ticker = "AAPL"
    
    print(f"\n📊 Fetching data for {ticker}...")
    fetch_start = time.time()
    df = fetch_stock_data(ticker, period="3mo")
    fetch_time = time.time() - fetch_start
    
    if df is None or df.empty:
        print(f"❌ Error: Could not fetch data for {ticker}")
        return
    
    print(f"✅ Fetched {len(df)} days of data in {fetch_time:.2f}s")
    
    # Calculate indicators
    print("\n🔧 Calculating technical indicators...")
    indicators_start = time.time()
    df_with_indicators = calculate_all_indicators(df)
    indicators_time = time.time() - indicators_start
    print(f"✅ Indicators calculated in {indicators_time:.2f}s")
    
    # Get signals
    print("\n📈 Extracting trading signals...")
    signals_start = time.time()
    signals = get_current_signals(df_with_indicators)
    signals_time = time.time() - signals_start
    print(f"✅ Signals extracted in {signals_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total processing time: {total_time:.2f}s\n")
    
    # Print formatted output
    print_signals(signals, ticker)


def example_full_scan_sequential():
    """Example 2: Run full scanner sequentially"""
    print("=" * 70)
    print("EXAMPLE 2: Full Scanner (Sequential Mode)")
    print("=" * 70)
    print("\nThis scans multiple stocks one at a time.\n")
    
    start_time = time.time()
    
    # Initialize scanner (sequential by default)
    print("🚀 Initializing scanner...")
    scanner = AgenticStockScanner(parallel=False)
    print("✅ Scanner initialized\n")
    
    # Scan stocks
    print("🔍 Starting stock scan (this may take a while)...")
    scan_start = time.time()
    results = scanner.scan_stocks(focus="all", period="3mo", parallel=False)
    scan_time = time.time() - scan_start
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Scan completed in {scan_time:.2f}s (Total: {total_time:.2f}s)")
    print(f"📊 Scanned {results.get('stocks_scanned', 0)} stocks")
    print(f"🎯 Found {results.get('buy_opportunities', 0)} buy opportunities\n")
    
    # Print recommendations
    scanner.print_recommendations(limit=10)


def example_full_scan_parallel():
    """Example 3: Run full scanner with parallel processing"""
    print("=" * 70)
    print("EXAMPLE 3: Full Scanner (Parallel Mode)")
    print("=" * 70)
    print("\nThis analyzes multiple stocks simultaneously for faster results.\n")
    
    start_time = time.time()
    
    # Initialize scanner with parallel processing
    print("🚀 Initializing scanner with parallel processing...")
    scanner = AgenticStockScanner(parallel=True, max_workers=5)
    print(f"✅ Scanner initialized with {scanner.max_workers} workers\n")
    
    # Scan stocks in parallel
    print("🔍 Starting parallel stock scan...")
    scan_start = time.time()
    results = scanner.scan_stocks(focus="all", period="3mo", parallel=True)
    scan_time = time.time() - scan_start
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Scan completed in {scan_time:.2f}s (Total: {total_time:.2f}s)")
    print(f"📊 Scanned {results.get('stocks_scanned', 0)} stocks")
    print(f"🎯 Found {results.get('buy_opportunities', 0)} buy opportunities")
    print(f"⚡ Parallel processing speedup: ~{scanner.max_workers}x faster\n")
    
    # Print recommendations
    scanner.print_recommendations(limit=10)


def example_custom_analysis(tickers: Optional[List[str]] = None):
    """Example 4: Custom analysis script template"""
    print("=" * 70)
    print("EXAMPLE 4: Custom Analysis Template")
    print("=" * 70)
    
    start_time = time.time()
    
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
    
    print(f"\n📊 Analyzing {len(custom_tickers)} custom ticker(s): {', '.join(custom_tickers)}\n")
    
    successful = 0
    failed = []
    
    # Use progress bar for ticker analysis
    for ticker in tqdm(custom_tickers, desc="Processing tickers", unit="ticker", ncols=100):
        ticker_start = time.time()
        
        # Fetch data
        df = fetch_stock_data(ticker, period="3mo")
        if df is None or df.empty:
            failed.append(ticker)
            tqdm.write(f"⚠️ Could not fetch {ticker}")
            continue
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(df)
        signals = get_current_signals(df_with_indicators)
        
        ticker_time = time.time() - ticker_start
        successful += 1
        
        # Print formatted (using tqdm.write to avoid interfering with progress bar)
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"✅ {ticker} analyzed in {ticker_time:.2f}s")
        tqdm.write('='*60)
        print_signals(signals, ticker)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"📊 Analysis Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully analyzed: {successful}/{len(custom_tickers)} tickers")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Average time per ticker: {total_time/len(custom_tickers):.2f}s")
    print(f"{'='*60}\n")


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

