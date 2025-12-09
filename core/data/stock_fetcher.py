"""
Stock Data Fetcher - Fetches stock data for tech stocks and rising stocks
"""
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
import time

# Tech stock tickers (major tech companies)
# Note: ANSS and SPLK removed due to data availability issues (404 errors)
TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'CSCO', 'AVGO', 'QCOM',
    'TXN', 'AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'FTNT',
    'PANW', 'CRWD', 'ZS', 'NET', 'DDOG', 'MDB', 'SNOW', 'PLTR',
    'RBLX', 'U', 'DOCN', 'GTLB', 'TEAM', 'ZM', 'OKTA'
]

# Rising stocks (can be updated dynamically)
RISING_STOCKS = [
    'SMCI', 'SOUN', 'ARM', 'RDDT', 'GME', 'AMC', 'SPY', 'QQQ',
    'TQQQ', 'SOXL', 'TECL', 'ARKK', 'ARKQ', 'ARKW'
]

def fetch_stock_data(ticker: str, period: str = "3mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch stock data for a given ticker
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return None
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None
        
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {str(e)}")
        return None

def fetch_multiple_stocks(tickers: List[str], period: str = "3mo", 
                         interval: str = "1d", delay: float = 0.1) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple stocks with rate limiting
    
    Args:
        tickers: List of stock tickers
        period: Time period
        interval: Data interval
        delay: Delay between requests (seconds)
    
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    results = {}
    
    for ticker in tickers:
        df = fetch_stock_data(ticker, period, interval)
        if df is not None and len(df) > 0:
            results[ticker] = df
        time.sleep(delay)  # Rate limiting
    
    return results

def get_tech_stocks(period: str = "3mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Fetch data for all tech stocks"""
    return fetch_multiple_stocks(TECH_STOCKS, period, interval)

def get_rising_stocks(period: str = "3mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Fetch data for rising stocks"""
    return fetch_multiple_stocks(RISING_STOCKS, period, interval)

def get_all_stocks(period: str = "3mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Fetch data for all tracked stocks (tech + rising)"""
    all_tickers = TECH_STOCKS + RISING_STOCKS
    return fetch_multiple_stocks(all_tickers, period, interval)

def get_stock_info(ticker: str) -> Dict:
    """Get additional stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'dividend_yield': info.get('dividendYield', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', None),
            '52_week_low': info.get('fiftyTwoWeekLow', None),
            'current_price': info.get('currentPrice', None),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
        }
    except Exception as e:
        print(f"Error fetching info for {ticker}: {str(e)}")
        return {'ticker': ticker, 'name': ticker}

if __name__ == "__main__":
    # Test the fetcher
    print("Fetching tech stocks...")
    tech_data = get_tech_stocks(period="1mo")
    print(f"Fetched {len(tech_data)} tech stocks")
    
    print("\nFetching rising stocks...")
    rising_data = get_rising_stocks(period="1mo")
    print(f"Fetched {len(rising_data)} rising stocks")
    
    # Show sample
    if tech_data:
        ticker = list(tech_data.keys())[0]
        print(f"\nSample data for {ticker}:")
        print(tech_data[ticker].tail())
