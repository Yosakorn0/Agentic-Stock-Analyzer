"""
Forex/Commodity Data Fetcher - Alternative data sources for XAUUSD and other forex pairs

Supports multiple data sources:
1. TradingView (via tvkit or pytradingview) - Most accurate, matches TradingView charts
2. OANDA API - Free tier, excellent for forex/commodities
3. Twelve Data - Free tier, good forex coverage
4. Alpha Vantage - Free tier, supports forex
5. Yahoo Finance (fallback) - For stocks/ETFs

Usage:
    from core.data.forex_fetcher import fetch_xauusd_data
    
    # Try TradingView first, fallback to others
    df = fetch_xauusd_data(period="60d", interval="5m")
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Try importing optional dependencies
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from pytradingview import TradingView
    HAS_PYTRADINGVIEW = True
except ImportError:
    HAS_PYTRADINGVIEW = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def _convert_period_to_tradingview(period: str) -> Dict:
    """
    Convert period string to TradingView format
    
    Args:
        period: Period string (e.g., "60d", "1mo", "5y")
    
    Returns:
        Dict with count and unit for TradingView
    """
    if period.endswith('d'):
        days = int(period[:-1])
        if days <= 1:
            return {'count': 1, 'unit': 'D'}
        elif days <= 7:
            return {'count': days, 'unit': 'D'}
        elif days <= 30:
            return {'count': days, 'unit': 'D'}
        else:
            months = days // 30
            return {'count': months, 'unit': 'M'}
    elif period.endswith('mo'):
        months = int(period[:-2])
        return {'count': months, 'unit': 'M'}
    elif period.endswith('y'):
        years = int(period[:-1])
        return {'count': years, 'unit': 'Y'}
    else:
        return {'count': 60, 'unit': 'D'}


def _convert_interval_to_tradingview(interval: str) -> str:
    """
    Convert interval string to TradingView format
    
    Args:
        interval: Interval string (e.g., "5m", "1h", "1d")
    
    Returns:
        TradingView interval string
    """
    mapping = {
        '1m': '1',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '4h': '240',
        '1d': 'D',
        '1w': 'W',
        '1mo': 'M'
    }
    return mapping.get(interval.lower(), '5')


def fetch_tradingview_xauusd(period: str = "60d", interval: str = "5m", timeout: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch XAUUSD data from TradingView using pytradingview
    
    Args:
        period: Time period (60d, 1mo, 3mo, 1y, etc.)
        interval: Data interval (5m, 15m, 1h, 1d, etc.)
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    if not HAS_PYTRADINGVIEW:
        return None
    
    try:
        tv = TradingView()
        
        # TradingView symbol for XAUUSD
        symbol = "XAUUSD"
        
        # Convert period and interval
        period_dict = _convert_period_to_tradingview(period)
        tv_interval = _convert_interval_to_tradingview(interval)
        
        # Fetch data with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                tv.get_historical,
                symbol=symbol,
                exchange="FX",
                interval=tv_interval,
                n_bars=1000  # Max bars to fetch
            )
            data = future.result(timeout=timeout)
        
        if not data or 'time' not in data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Standardize column names
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        return df
        
    except Exception as e:
        print(f"TradingView fetch error: {str(e)[:100]}")
        return None


def fetch_oanda_xauusd(period: str = "60d", interval: str = "5m", api_key: Optional[str] = None, timeout: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch XAUUSD data from OANDA API (requires free API key)
    
    Args:
        period: Time period (60d, 1mo, 3mo, 1y, etc.)
        interval: Data interval (5m, 15m, 1h, 1d, etc.)
        api_key: OANDA API key (or from OANDA_API_KEY env var)
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    if not HAS_REQUESTS:
        return None
    
    # Get API key from environment or parameter
    api_key = api_key or os.getenv('OANDA_API_KEY')
    if not api_key:
        return None  # No API key available
    
    try:
        # OANDA API endpoint
        account_id = os.getenv('OANDA_ACCOUNT_ID', '')
        base_url = "https://api-fxpractice.oanda.com"  # Practice account
        
        # Convert interval to OANDA format
        interval_mapping = {
            '1m': 'M1',
            '5m': 'M5',
            '15m': 'M15',
            '30m': 'M30',
            '1h': 'H1',
            '4h': 'H4',
            '1d': 'D'
        }
        oanda_granularity = interval_mapping.get(interval.lower(), 'M5')
        
        # Calculate date range
        if period.endswith('d'):
            days = int(period[:-1])
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
        elif period.endswith('mo'):
            months = int(period[:-2])
            end_time = datetime.now()
            start_time = end_time - timedelta(days=months * 30)
        elif period.endswith('y'):
            years = int(period[:-1])
            end_time = datetime.now()
            start_time = end_time - timedelta(days=years * 365)
        else:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=60)
        
        # OANDA API request
        url = f"{base_url}/v3/instruments/XAU_USD/candles"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        params = {
            'granularity': oanda_granularity,
            'from': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'to': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'count': 5000  # Max candles
        }
        
        # Fetch with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(requests.get, url, headers=headers, params=params)
            response = future.result(timeout=timeout)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        if 'candles' not in data:
            return None
        
        # Convert to DataFrame
        candles = []
        for candle in data['candles']:
            if candle['complete']:  # Only use completed candles
                candles.append({
                    'time': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
        
        if not candles:
            return None
        
        df = pd.DataFrame(candles)
        df.set_index('time', inplace=True)
        df.columns = [col.lower() for col in df.columns]
        
        return df
        
    except Exception as e:
        print(f"OANDA fetch error: {str(e)[:100]}")
        return None


def fetch_yahoo_fallback(ticker: str, period: str = "60d", interval: str = "5m", timeout: int = 30) -> Optional[pd.DataFrame]:
    """
    Fallback to Yahoo Finance (for GC=F, GLD, etc.)
    
    Args:
        ticker: Ticker symbol (e.g., "GC=F", "GLD")
        period: Time period
        interval: Data interval
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    if not HAS_YFINANCE:
        return None
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: yf.Ticker(ticker).history(period=period, interval=interval)
            )
            df = future.result(timeout=timeout)
        
        if df.empty:
            return None
        
        df.columns = [col.lower() for col in df.columns]
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None
        
        return df
        
    except Exception as e:
        print(f"Yahoo Finance fetch error: {str(e)[:100]}")
        return None


def fetch_xauusd_data(
    period: str = "60d",
    interval: str = "5m",
    source: Optional[str] = None,
    timeout: int = 30
) -> Optional[pd.DataFrame]:
    """
    Fetch XAUUSD data from the best available source
    
    Priority order:
    1. TradingView (if pytradingview installed) - Most accurate, matches TradingView charts
    2. OANDA (if API key configured) - Professional forex data
    3. Yahoo Finance GC=F (fallback) - May have limitations
    
    Args:
        period: Time period (60d, 1mo, 3mo, 1y, etc.)
        interval: Data interval (5m, 15m, 1h, 1d, etc.)
        source: Force specific source ('tradingview', 'oanda', 'yahoo') or None for auto
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with OHLCV data or None if all sources fail
    """
    if source == 'tradingview' or source is None:
        df = fetch_tradingview_xauusd(period=period, interval=interval, timeout=timeout)
        if df is not None and not df.empty:
            return df
    
    if source == 'oanda' or (source is None and os.getenv('OANDA_API_KEY')):
        df = fetch_oanda_xauusd(period=period, interval=interval, timeout=timeout)
        if df is not None and not df.empty:
            return df
    
    # Fallback to Yahoo Finance
    if source == 'yahoo' or source is None:
        # Try GC=F (Gold Futures) as fallback
        for fallback_ticker in ['GC=F', 'GLD']:
            df = fetch_yahoo_fallback(fallback_ticker, period=period, interval=interval, timeout=timeout)
            if df is not None and not df.empty:
                return df
    
    return None


if __name__ == "__main__":
    # Test the fetcher
    print("Testing XAUUSD data fetcher...")
    print("\n1. Trying TradingView...")
    df = fetch_tradingview_xauusd(period="5d", interval="5m")
    if df is not None:
        print(f"✅ TradingView: {len(df)} bars")
        print(df.tail())
    else:
        print("❌ TradingView: Not available (install: pip install pytradingview)")
    
    print("\n2. Trying OANDA...")
    df = fetch_oanda_xauusd(period="5d", interval="5m")
    if df is not None:
        print(f"✅ OANDA: {len(df)} bars")
        print(df.tail())
    else:
        print("❌ OANDA: Not available (set OANDA_API_KEY env var)")
    
    print("\n3. Trying Yahoo Finance fallback...")
    df = fetch_yahoo_fallback("GC=F", period="5d", interval="5m")
    if df is not None:
        print(f"✅ Yahoo Finance: {len(df)} bars")
        print(df.tail())
    else:
        print("❌ Yahoo Finance: Failed")
    
    print("\n4. Auto-detect best source...")
    df = fetch_xauusd_data(period="5d", interval="5m")
    if df is not None:
        print(f"✅ Auto-detect: {len(df)} bars")
        print(df.tail())
    else:
        print("❌ Auto-detect: All sources failed")

