"""
Train and backtest ML model for a single stock

This script:
1. Trains an ML model using data for ONE ticker (e.g. AAPL)
2. Backtests that single-stock ML strategy

Usage:
    python examples/train_and_backtest_single.py --ticker AAPL
"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis.ml_analyzer import MLStockAnalyzer
from core.backtesting.backtester import Backtester
from core.data.stock_fetcher import fetch_stock_data
from core.analysis.technical_analyzer import calculate_all_indicators, get_current_signals
from datetime import datetime, timedelta


def train_ml_model_single(ticker: str):
    """Train ML model on historical data for a single ticker"""
    print("=" * 80)
    print(f"🤖 TRAINING ML MODEL FOR {ticker}")
    print("=" * 80)

    # Fetch historical data (1 year for training)
    print(f"📊 Fetching historical data for {ticker} (1y, 1d)...")
    df = fetch_stock_data(ticker, period="1y", interval="1d")

    if df is None or df.empty:
        print(f"❌ No data fetched for {ticker}")
        return None, None

    from core.analysis.technical_analyzer import calculate_all_indicators

    # Prepare dict in same format as multi-stock training
    stocks_data = {ticker: df}

    # Train model
    ml_analyzer = MLStockAnalyzer(model_type="random_forest")
    try:
        metrics = ml_analyzer.train_model(stocks_data, forward_periods=5)

        # Save model with ticker-specific name
        model_path = f"models/stock_predictor_{ticker}.pkl"
        ml_analyzer.save_model(model_path)

        print(f"✅ Trained and saved model for {ticker} -> {model_path}")
        return ml_analyzer, metrics
    except Exception as e:
        print(f"❌ Error training model for {ticker}: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None


def backtest_ml_strategy_single(ticker: str, ml_analyzer: MLStockAnalyzer):
    """Backtest the ML strategy for a single ticker"""
    print("\n" + "=" * 80)
    print(f"📈 BACKTESTING ML STRATEGY FOR {ticker}")
    print("=" * 80)

    # Fetch test data (last 6 months)
    print(f"📊 Fetching test data for {ticker} (6mo, 1d)...")
    df = fetch_stock_data(ticker, period="6mo", interval="1d")

    if df is None or df.empty:
        print(f"❌ No test data available for {ticker}")
        return None

    from core.analysis.technical_analyzer import calculate_all_indicators

    # Calculate indicators
    df_indicators = calculate_all_indicators(df)
    if len(df_indicators) < 50:
        print(f"❌ Not enough data for backtest ({len(df_indicators)} rows, need >= 50)")
        return None

    # Generate signals for each day
    signals = {}
    ticker_signals = {}

    print("🔄 Generating ML predictions for each day...")
    for i in range(50, len(df_indicators)):
        date = df_indicators.index[i]
        current_data = df_indicators.iloc[: i + 1]

        try:
            prediction = ml_analyzer.predict(current_data)
            ticker_signals[date] = prediction
        except Exception:
            continue

    if not ticker_signals:
        print("❌ No signals generated")
        return None

    signals[ticker] = ticker_signals

    # Run backtest
    backtester = Backtester(initial_capital=10000, commission=0.001)
    results = backtester.backtest_strategy({ticker: df}, signals, analyzer_type="ml")

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/ml_backtest_results_{ticker}.json"
    backtester.save_results(results, results_path)

    print(f"✅ Saved backtest results to {results_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train and backtest ML model for a single stock")
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker to train/backtest on (default: AAPL)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    print("=" * 80)
    print(f"🚀 ML TRAIN & BACKTEST (SINGLE STOCK: {ticker})")
    print("=" * 80)
    print()

    # Train ML model for single ticker
    ml_analyzer, metrics = train_ml_model_single(ticker)

    if ml_analyzer is None:
        print("\n❌ Failed to train model. Exiting.")
        return

    # Backtest ML strategy
    results = backtest_ml_strategy_single(ticker, ml_analyzer)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    if metrics:
        print(f"\n🤖 ML MODEL METRICS ({ticker}):")
        print(f"   Test R²: {metrics['test_r2']:.3f}")
        print(f"   Test MAE: {metrics['test_mae']:.2f}%")
        print(f"   CV MAE: {metrics['cv_mae']:.2f}%")

    if results:
        print(f"\n📈 ML STRATEGY BACKTEST ({ticker}):")
        print(f"   Total Return: {results['total_return']:.2f}%")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {results.get('max_drawdown', 0):.2f}%")

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


