"""
Train ML model and backtest strategy

This script:
1. Trains an ML model on historical stock data
2. Backtests the ML model predictions
3. Compares performance with technical analysis

Usage:
    python examples/train_and_backtest.py
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis.ml_analyzer import MLStockAnalyzer
from core.backtesting.backtester import Backtester
from core.data.stock_fetcher import get_tech_stocks, fetch_stock_data
from core.analysis.technical_analyzer import calculate_all_indicators, get_current_signals
from core.analysis.ai_analyzer import StockAIAnalyzer
from datetime import datetime, timedelta


def train_ml_model():
    """Train ML model on historical data"""
    print("=" * 80)
    print("🤖 TRAINING ML MODEL")
    print("=" * 80)
    
    # Fetch historical data (1 year for training)
    print("📊 Fetching historical data for training...")
    stocks_data = get_tech_stocks(period="1y", interval="1d")
    
    if not stocks_data:
        print("❌ No stock data fetched")
        return None, None
    
    print(f"✅ Fetched data for {len(stocks_data)} stocks")
    
    # Train model
    ml_analyzer = MLStockAnalyzer(model_type="random_forest")
    try:
        metrics = ml_analyzer.train_model(stocks_data, forward_periods=5)
        
        # Save model
        ml_analyzer.save_model("models/stock_predictor.pkl")
        
        return ml_analyzer, metrics
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def backtest_ml_strategy(ml_analyzer):
    """Backtest the ML strategy"""
    print("\n" + "=" * 80)
    print("📈 BACKTESTING ML STRATEGY")
    print("=" * 80)
    
    # Fetch test data (last 6 months)
    print("📊 Fetching test data...")
    test_stocks = get_tech_stocks(period="6mo", interval="1d")
    
    if not test_stocks:
        print("❌ No test data available")
        return None
    
    # Generate signals for each day
    signals = {}
    print("🔄 Generating ML predictions for each day...")
    
    for ticker, df in test_stocks.items():
        try:
            df_indicators = calculate_all_indicators(df)
            ticker_signals = {}
            
            # Need at least 50 days of data for prediction
            if len(df_indicators) < 50:
                continue
            
            for i in range(50, len(df_indicators)):
                date = df_indicators.index[i]
                current_data = df_indicators.iloc[:i+1]
                
                # Get ML prediction
                try:
                    prediction = ml_analyzer.predict(current_data)
                    ticker_signals[date] = prediction
                except Exception as e:
                    continue
            
            if ticker_signals:
                signals[ticker] = ticker_signals
        except Exception as e:
            print(f"   ⚠️  Skipping {ticker}: {str(e)[:100]}")
            continue
    
    if not signals:
        print("❌ No signals generated")
        return None
    
    print(f"✅ Generated signals for {len(signals)} stocks")
    
    # Run backtest
    backtester = Backtester(initial_capital=10000, commission=0.001)
    results = backtester.backtest_strategy(test_stocks, signals, analyzer_type="ml")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    backtester.save_results(results, "results/ml_backtest_results.json")
    
    return results


def backtest_technical_strategy():
    """Backtest technical analysis strategy for comparison"""
    print("\n" + "=" * 80)
    print("📈 BACKTESTING TECHNICAL ANALYSIS STRATEGY")
    print("=" * 80)
    
    # Fetch test data
    print("📊 Fetching test data...")
    test_stocks = get_tech_stocks(period="6mo", interval="1d")
    
    if not test_stocks:
        return None
    
    # Generate signals based on technical analysis
    signals = {}
    print("🔄 Generating technical analysis signals...")
    
    for ticker, df in test_stocks.items():
        try:
            df_indicators = calculate_all_indicators(df)
            ticker_signals = {}
            
            for date in df_indicators.index:
                current_data = df_indicators.loc[:date]
                if len(current_data) < 20:
                    continue
                
                # Get technical signals
                tech_signals = get_current_signals(current_data)
                
                # Convert to recommendation format
                technical_score = 50
                rsi = tech_signals.get('rsi', 50)
                trend = tech_signals.get('direction', 'unknown')
                change_5d = tech_signals.get('price_change_5d', 0)
                
                # Simple rule-based recommendation
                if trend == 'up' and rsi < 70 and change_5d > 0:
                    recommendation = "BUY"
                    confidence = min(75, 60 + int(change_5d))
                elif rsi < 30:
                    recommendation = "BUY"
                    confidence = 70
                elif trend == 'down' and rsi > 70:
                    recommendation = "SELL"
                    confidence = 60
                else:
                    recommendation = "WAIT"
                    confidence = 50
                
                ticker_signals[date] = {
                    'recommendation': recommendation,
                    'confidence': confidence
                }
            
            signals[ticker] = ticker_signals
        except Exception as e:
            print(f"   ⚠️  Skipping {ticker}: {str(e)[:100]}")
            continue
    
    if not signals:
        return None
    
    # Run backtest
    backtester = Backtester(initial_capital=10000, commission=0.001)
    results = backtester.backtest_strategy(test_stocks, signals, analyzer_type="technical")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    backtester.save_results(results, "results/technical_backtest_results.json")
    
    return results


def main():
    """Main function"""
    print("=" * 80)
    print("🚀 ML MODEL TRAINING & BACKTESTING")
    print("=" * 80)
    print()
    
    # Train ML model
    ml_analyzer, metrics = train_ml_model()
    
    if ml_analyzer is None:
        print("\n❌ Failed to train model. Exiting.")
        return
    
    # Backtest ML strategy
    ml_results = backtest_ml_strategy(ml_analyzer)
    
    # Backtest technical strategy for comparison
    tech_results = backtest_technical_strategy()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    if metrics:
        print(f"\n🤖 ML MODEL METRICS:")
        print(f"   Test R²: {metrics['test_r2']:.3f}")
        print(f"   Test MAE: {metrics['test_mae']:.2f}%")
        print(f"   CV MAE: {metrics['cv_mae']:.2f}%")
    
    if ml_results:
        print(f"\n📈 ML STRATEGY BACKTEST:")
        print(f"   Total Return: {ml_results['total_return']:.2f}%")
        print(f"   Total Trades: {ml_results['total_trades']}")
        print(f"   Win Rate: {ml_results['win_rate']:.1f}%")
        print(f"   Profit Factor: {ml_results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {ml_results['max_drawdown']:.2f}%")
    
    if tech_results:
        print(f"\n📊 TECHNICAL ANALYSIS BACKTEST:")
        print(f"   Total Return: {tech_results['total_return']:.2f}%")
        print(f"   Total Trades: {tech_results['total_trades']}")
        print(f"   Win Rate: {tech_results['win_rate']:.1f}%")
        print(f"   Profit Factor: {tech_results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {tech_results['max_drawdown']:.2f}%")
    
    if ml_results and tech_results:
        print(f"\n🏆 COMPARISON:")
        if ml_results['total_return'] > tech_results['total_return']:
            print(f"   ✅ ML Strategy outperformed by {ml_results['total_return'] - tech_results['total_return']:.2f}%")
        else:
            print(f"   ✅ Technical Strategy outperformed by {tech_results['total_return'] - ml_results['total_return']:.2f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

