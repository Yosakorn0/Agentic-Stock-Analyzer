"""
Agentic AI Stock Scanner - Unified orchestrator with optional parallel processing

Supports both sequential and parallel processing modes.
"""
import time
from typing import Dict, List, Optional
from datetime import datetime
import json

# Optional parallel processing support
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

from core.data import get_all_stocks, get_stock_info
from core.analysis import calculate_all_indicators, get_current_signals
from core.analysis import StockAIAnalyzer
from core.screening import StockScreener


class AgenticStockScanner:
    """Main agentic AI system for scanning and analyzing stocks
    
    Supports both sequential and parallel processing modes.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini", 
                 max_workers: int = 5, parallel: bool = False):
        """
        Initialize the agentic scanner
        
        Args:
            openai_api_key: OpenAI API key (optional, can use env var)
            model: AI model to use
            max_workers: Maximum number of parallel workers (only used if parallel=True)
            parallel: Enable parallel processing (faster but uses more API quota)
        """
        self.ai_analyzer = StockAIAnalyzer(api_key=openai_api_key, model=model)
        self.screener = StockScreener()
        self.scan_results = {}
        self.max_workers = max_workers
        self.parallel = parallel and PARALLEL_AVAILABLE
        
        if parallel and not PARALLEL_AVAILABLE:
            print("⚠️ Warning: Parallel processing requested but concurrent.futures not available. Using sequential mode.")
    
    def _analyze_single_stock(self, ticker: str, stock_info: Dict, 
                             technical_signals: Dict, price_summary: Dict) -> tuple:
        """Analyze a single stock (used for parallel processing)"""
        try:
            # Create a new analyzer instance for thread safety
            analyzer = StockAIAnalyzer(
                api_key=self.ai_analyzer.api_key,
                model=self.ai_analyzer.model
            )
            
            analysis = analyzer.analyze_stock(
                ticker=ticker,
                stock_info=stock_info,
                technical_signals=technical_signals,
                price_data_summary=price_summary
            )
            
            return (ticker, analysis)
        except Exception as e:
            print(f"  ⚠️ Error analyzing {ticker}: {str(e)}")
            return (ticker, {
                'recommendation': 'ERROR',
                'confidence': 0,
                'reasoning': f"Error: {str(e)}",
                'upside_potential': 'Unknown',
                'risk_level': 'High',
                'technical_score': 0,
                'ai_analysis': ''
            })
    
    def scan_stocks(self, tickers: Optional[List[str]] = None,
                   period: str = "3mo", interval: str = "1d",
                   focus: str = "all", parallel: Optional[bool] = None) -> Dict:
        """
        Scan stocks and generate buy recommendations
        
        Args:
            tickers: List of specific tickers to scan (None = scan all)
            period: Time period for data
            interval: Data interval
            focus: Focus area ("tech", "rising", "all")
            parallel: Override parallel setting (None = use instance default)
        
        Returns:
            Dictionary with scan results
        """
        use_parallel = parallel if parallel is not None else self.parallel
        
        mode_str = "PARALLEL" if use_parallel else "SEQUENTIAL"
        print(f"\n{'='*60}")
        print(f"AGENTIC AI STOCK SCANNER ({mode_str}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Fetch stock data
        print("📊 Fetching stock data...")
        if tickers:
            from core.data import fetch_multiple_stocks
            stocks_data = fetch_multiple_stocks(tickers, period, interval)
        else:
            if focus == "tech":
                from core.data import get_tech_stocks
                stocks_data = get_tech_stocks(period, interval)
            elif focus == "rising":
                from core.data import get_rising_stocks
                stocks_data = get_rising_stocks(period, interval)
            else:
                stocks_data = get_all_stocks(period, interval)
        
        print(f"✅ Fetched data for {len(stocks_data)} stocks\n")
        
        if not stocks_data:
            return {'error': 'No stock data fetched', 'recommendations': []}
        
        # Get stock information
        print("📋 Gathering stock information...")
        stock_infos = {}
        
        if use_parallel and PARALLEL_AVAILABLE:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(get_stock_info, ticker): ticker 
                          for ticker in stocks_data.keys()}
                
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        info = future.result()
                        stock_infos[ticker] = info
                        print(f"  ✅ {ticker}")
                    except Exception as e:
                        print(f"  ⚠️ {ticker}: {str(e)}")
                        stock_infos[ticker] = {'ticker': ticker, 'name': ticker}
        else:
            for ticker in stocks_data.keys():
                info = get_stock_info(ticker)
                stock_infos[ticker] = info
                time.sleep(0.1)  # Rate limiting
        
        print(f"✅ Collected info for {len(stock_infos)} stocks\n")
        
        # Calculate technical indicators
        print("🔧 Calculating technical indicators...")
        stocks_with_indicators = {}
        technical_signals = {}
        
        def process_indicators(ticker: str, df) -> tuple:
            df_indicators = calculate_all_indicators(df)
            signals = get_current_signals(df_indicators)
            return (ticker, df_indicators, signals)
        
        if use_parallel and PARALLEL_AVAILABLE:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_indicators, ticker, df): ticker 
                          for ticker, df in stocks_data.items()}
                
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        ticker_result, df_indicators, signals = future.result()
                        stocks_with_indicators[ticker_result] = df_indicators
                        technical_signals[ticker_result] = signals
                        print(f"  ✅ {ticker_result}")
                    except Exception as e:
                        print(f"  ⚠️ {ticker}: {str(e)}")
        else:
            for ticker, df in stocks_data.items():
                df_indicators = calculate_all_indicators(df)
                stocks_with_indicators[ticker] = df_indicators
                signals = get_current_signals(df_indicators)
                technical_signals[ticker] = signals
        
        print(f"✅ Calculated indicators for {len(stocks_with_indicators)} stocks\n")
        
        # AI Analysis
        print(f"🤖 Running AI analysis ({'parallel' if use_parallel else 'sequential'})...")
        analysis_results = {}
        
        # Prepare analysis tasks
        analysis_tasks = []
        for ticker in stocks_data.keys():
            stock_info = stock_infos.get(ticker, {})
            signals = technical_signals.get(ticker, {})
            
            price_summary = {
                'current_price': signals.get('current_price', 0),
                'price_change_1d': signals.get('price_change_1d', 0),
                'price_change_5d': signals.get('price_change_5d', 0),
                'price_change_20d': signals.get('price_change_20d', 0),
            }
            
            analysis_tasks.append((ticker, stock_info, signals, price_summary))
        
        # Execute analysis
        if use_parallel and PARALLEL_AVAILABLE:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._analyze_single_stock, ticker, info, signals, price): ticker
                    for ticker, info, signals, price in analysis_tasks
                }
                
                completed = 0
                total = len(futures)
                
                for future in as_completed(futures):
                    ticker = futures[future]
                    completed += 1
                    try:
                        ticker_result, analysis = future.result()
                        analysis_results[ticker_result] = analysis
                        recommendation = analysis.get('recommendation', 'WAIT')
                        confidence = analysis.get('confidence', 0)
                        
                        # Choose icon based on recommendation
                        if recommendation == 'BUY':
                            icon = '🟢'
                        elif recommendation == 'CONSIDER BUY':
                            icon = '🟡'
                        elif recommendation == 'WATCH':
                            icon = '🟠'
                        else:  # WAIT
                            icon = '🔴'
                        
                        print(f"  [{completed}/{total}] {ticker_result}: {icon} {recommendation} ({confidence}%)")
                    except Exception as e:
                        print(f"  ⚠️ Error with {ticker}: {str(e)}")
        else:
            # Sequential processing
            for ticker, stock_info, signals, price_summary in analysis_tasks:
                print(f"  Analyzing {ticker}...", end=" ")
                analysis = self.ai_analyzer.analyze_stock(
                    ticker=ticker,
                    stock_info=stock_info,
                    technical_signals=signals,
                    price_data_summary=price_summary
                )
                analysis_results[ticker] = analysis
                
                # Choose icon based on recommendation
                recommendation = analysis.get('recommendation', 'WAIT')
                confidence = analysis.get('confidence', 0)
                
                if recommendation == 'BUY':
                    icon = '🟢'
                elif recommendation == 'CONSIDER BUY':
                    icon = '🟡'
                elif recommendation == 'WATCH':
                    icon = '🟠'
                else:  # WAIT
                    icon = '🔴'
                
                print(f"{icon} {recommendation} ({confidence}%)")
                time.sleep(0.2)  # Rate limiting for API
        
        print(f"\n✅ Completed AI analysis for {len(analysis_results)} stocks\n")
        
        # Screen for buy opportunities
        print("🔍 Screening for buy opportunities...")
        buy_opportunities = self.screener.screen_buy_opportunities(
            stocks_data, stock_infos, analysis_results
        )
        
        # Filter by confidence
        high_confidence_buys = []
        for ticker in buy_opportunities:
            if analysis_results[ticker].get('confidence', 0) >= 70:
                high_confidence_buys.append(ticker)
        
        print(f"✅ Found {len(buy_opportunities)} buy opportunities")
        print(f"✅ Found {len(high_confidence_buys)} high-confidence buys (≥70%)\n")
        
        # Compile results
        results = {
            'scan_timestamp': datetime.now().isoformat(),
            'stocks_scanned': len(stocks_data),
            'buy_opportunities': len(buy_opportunities),
            'high_confidence_buys': len(high_confidence_buys),
            'recommendations': self._compile_recommendations(
                buy_opportunities, analysis_results, stock_infos, technical_signals
            ),
            'all_analysis': analysis_results,
            'parallel_mode': use_parallel
        }
        
        self.scan_results = results
        return results
    
    def _compile_recommendations(self, buy_tickers: List[str],
                                analysis_results: Dict,
                                stock_infos: Dict,
                                technical_signals: Dict) -> List[Dict]:
        """Compile buy recommendations with all relevant data"""
        recommendations = []
        
        for ticker in buy_tickers:
            analysis = analysis_results.get(ticker, {})
            info = stock_infos.get(ticker, {})
            signals = technical_signals.get(ticker, {})
            
            recommendation = {
                'ticker': ticker,
                'name': info.get('name', ticker),
                'sector': info.get('sector', 'Unknown'),
                'current_price': signals.get('current_price', 0),
                'recommendation': analysis.get('recommendation', 'WAIT'),
                'confidence': analysis.get('confidence', 0),
                'upside_potential': analysis.get('upside_potential', 'Medium'),
                'risk_level': analysis.get('risk_level', 'Medium'),
                'reasoning': analysis.get('reasoning', ''),
                'technical_score': analysis.get('technical_score', 0),
                'price_change_1d': signals.get('price_change_1d', 0),
                'price_change_5d': signals.get('price_change_5d', 0),
                'price_change_20d': signals.get('price_change_20d', 0),
                'trend': signals.get('direction', 'unknown'),
                'rsi': signals.get('rsi', 50),
            }
            
            recommendations.append(recommendation)
        
        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return recommendations
    
    def print_recommendations(self, limit: int = 10):
        """Print formatted buy recommendations"""
        if not self.scan_results or 'recommendations' not in self.scan_results:
            print("No recommendations available. Run scan_stocks() first.")
            return
        
        recommendations = self.scan_results['recommendations'][:limit]
        
        print(f"\n{'='*80}")
        print(f"TOP {len(recommendations)} BUY RECOMMENDATIONS")
        if self.scan_results.get('parallel_mode'):
            print("(Parallel Processing Mode)")
        print(f"{'='*80}\n")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['ticker']} - {rec['name']}")
            print(f"   Price: ${rec['current_price']:.2f} | Confidence: {rec['confidence']:.0f}%")
            print(f"   Upside: {rec['upside_potential']} | Risk: {rec['risk_level']}")
            print(f"   Change (5d): {rec['price_change_5d']:+.2f}% | Change (20d): {rec['price_change_20d']:+.2f}%")
            print(f"   Trend: {rec['trend']} | RSI: {rec['rsi']:.1f}")
            print(f"   Reasoning: {rec['reasoning'][:100]}...")
            print()
    
    def save_results(self, filename: Optional[str] = None):
        """Save scan results to JSON file"""
        if not self.scan_results:
            print("No results to save. Run scan_stocks() first.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scan_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.scan_results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agentic AI Stock Scanner')
    parser.add_argument('--focus', choices=['tech', 'rising', 'all'], default='all',
                       help='Focus area for scanning')
    parser.add_argument('--period', default='3mo', help='Time period for data')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--limit', type=int, default=10, help='Number of recommendations to show')
    parser.add_argument('--save', action='store_true', help='Save results to JSON file')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing (faster)')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = AgenticStockScanner(
        openai_api_key=args.api_key,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    # Run scan
    results = scanner.scan_stocks(focus=args.focus, period=args.period)
    
    # Print recommendations
    scanner.print_recommendations(limit=args.limit)
    
    # Save if requested
    if args.save:
        scanner.save_results()
    
    return results

if __name__ == "__main__":
    main()
