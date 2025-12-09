from core.analysis import get_current_signals, calculate_all_indicators
from core.data import fetch_stock_data
from utils.format_signals import print_signals

# Get ticker from user input
ticker = input("Enter stock ticker symbol (e.g., AAPL, MSFT, JEPQ): ").strip().upper()

if not ticker:
    print("⚠️ No ticker provided. Using default: AAPL")
    ticker = "AAPL"

# Fetch and analyze
print(f"\nAnalyzing {ticker}...")
df = fetch_stock_data(ticker, period="3mo")

if df is None or df.empty:
    print(f"Error: Could not fetch data for {ticker}")
    exit(1)

df_with_indicators = calculate_all_indicators(df)
signals = get_current_signals(df_with_indicators)

# Print formatted output
print_signals(signals, ticker)