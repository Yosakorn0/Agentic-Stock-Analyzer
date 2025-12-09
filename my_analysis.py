from core.analysis import get_current_signals, calculate_all_indicators
from core.data import fetch_stock_data


df = fetch_stock_data("MSFT", period="3mo")
df_with_indicators = calculate_all_indicators(df)
signals = get_current_signals(df_with_indicators)
print(signals)