"""
Agentic AI Stock Scanner - Complete Combined Version
Combines all modules into a single file for easy execution
"""
import os
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import argparse

# External library imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance library not installed. Install with: pip install yfinance")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")

# ============================================================================
# TECHNICAL ANALYZER FUNCTIONS
# ============================================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD indicator"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Momentum"""
    return series - series.shift(period)

def calculate_price_change(df: pd.DataFrame, periods: list = [1, 5, 10, 20]) -> Dict[str, pd.Series]:
    """Calculate price changes over multiple periods"""
    changes = {}
    for period in periods:
        changes[f'change_{period}d'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    return changes

def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate volume-based indicators"""
    if 'volume' not in df.columns:
        return {}
    
    indicators = {}
    indicators['volume_sma'] = calculate_sma(df['volume'], 20)
    indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
    
    return indicators

def analyze_trend(df: pd.DataFrame) -> Dict:
    """Analyze trend direction and strength"""
    if len(df) < 50:
        return {'direction': 'unknown', 'strength': 0, 'ema_trend': 'unknown'}
    
    close = df['close']
    ema_9 = calculate_ema(close, 9)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    last_idx = len(df) - 1
    
    # Trend direction
    if ema_9.iloc[last_idx] > ema_21.iloc[last_idx] > ema_50.iloc[last_idx]:
        direction = 'up'
    elif ema_9.iloc[last_idx] < ema_21.iloc[last_idx] < ema_50.iloc[last_idx]:
        direction = 'down'
    else:
        direction = 'sideways'
    
    # Trend strength (0-100)
    price_change_20d = ((close.iloc[last_idx] - close.iloc[last_idx-20]) / close.iloc[last_idx-20]) * 100 if last_idx >= 20 else 0
    strength = min(100, max(0, abs(price_change_20d) * 2))
    
    return {
        'direction': direction,
        'strength': strength,
        'ema_trend': direction,
        'price_change_20d': price_change_20d
    }

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    if df.empty or len(df) < 20:
        return df
    
    result_df = df.copy()
    close = result_df['close']
    
    # RSI
    result_df['rsi'] = calculate_rsi(close, 14)
    
    # Moving Averages
    result_df['ema_9'] = calculate_ema(close, 9)
    result_df['ema_21'] = calculate_ema(close, 21)
    result_df['ema_50'] = calculate_ema(close, 50)
    result_df['sma_20'] = calculate_sma(close, 20)
    result_df['sma_50'] = calculate_sma(close, 50)
    
    # MACD
    macd_data = calculate_macd(close)
    result_df['macd'] = macd_data['macd']
    result_df['macd_signal'] = macd_data['signal']
    result_df['macd_histogram'] = macd_data['histogram']
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(close)
    result_df['bb_upper'] = bb_data['upper']
    result_df['bb_middle'] = bb_data['middle']
    result_df['bb_lower'] = bb_data['lower']
    
    # ATR
    result_df['atr'] = calculate_atr(result_df, 14)
    
    # Momentum
    result_df['momentum'] = calculate_momentum(close, 10)
    
    # Price changes
    price_changes = calculate_price_change(result_df)
    for key, value in price_changes.items():
        result_df[key] = value
    
    # Volume indicators
    volume_indicators = calculate_volume_indicators(result_df)
    for key, value in volume_indicators.items():
        result_df[key] = value
    
    return result_df

def get_current_signals(df: pd.DataFrame) -> Dict:
    """Get current trading signals from indicators"""
    if df.empty or len(df) < 20:
        return {}
    
    last_idx = len(df) - 1
    last = df.iloc[last_idx]
    
    signals = {
        'rsi': last.get('rsi', 50),
        'rsi_signal': 'oversold' if last.get('rsi', 50) < 30 else ('overbought' if last.get('rsi', 50) > 70 else 'neutral'),
        'ema_cross': 'bullish' if last.get('ema_9', 0) > last.get('ema_21', 0) else 'bearish',
        'macd_signal': 'bullish' if last.get('macd', 0) > last.get('macd_signal', 0) else 'bearish',
        'bb_position': 'upper' if last['close'] > last.get('bb_upper', last['close']) else ('lower' if last['close'] < last.get('bb_lower', last['close']) else 'middle'),
        'price_change_1d': last.get('change_1d', 0),
        'price_change_5d': last.get('change_5d', 0),
        'price_change_20d': last.get('change_20d', 0),
        'current_price': last['close'],
    }
    
    # Trend analysis
    trend = analyze_trend(df)
    signals.update(trend)
    
    return signals

# ============================================================================
# STOCK FETCHER FUNCTIONS
# ============================================================================

# Tech stock tickers (major tech companies)
# Note: ANSS and SPLK removed due to data availability issues
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
    if not YFINANCE_AVAILABLE:
        print(f"Error: yfinance not available. Cannot fetch {ticker}")
        return None
    
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
    if not YFINANCE_AVAILABLE:
        return {'ticker': ticker, 'name': ticker}
    
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

# ============================================================================
# AI ANALYZER CLASS
# ============================================================================

class StockAIAnalyzer:
    """AI-powered stock analyzer using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize AI analyzer
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4o-mini, gpt-4, gpt-3.5-turbo)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
        elif not OPENAI_AVAILABLE:
            print("Warning: OpenAI library not available. AI analysis will be limited.")
    
    def analyze_stock(self, ticker: str, stock_info: Dict, technical_signals: Dict, 
                     price_data_summary: Dict) -> Dict:
        """
        Use AI to analyze a stock and generate buy recommendation
        
        Args:
            ticker: Stock ticker
            stock_info: Stock information (name, sector, etc.)
            technical_signals: Technical analysis signals
            price_data_summary: Summary of price data
        
        Returns:
            Dictionary with AI analysis and recommendation
        """
        if not self.client:
            # Fallback analysis without AI
            return self._fallback_analysis(ticker, stock_info, technical_signals, price_data_summary)
        
        try:
            # Prepare context for AI
            context = self._prepare_context(ticker, stock_info, technical_signals, price_data_summary)
            
            # Create prompt
            prompt = self._create_analysis_prompt(context)
            
            # Call AI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI response
            analysis = self._parse_ai_response(ai_response, technical_signals)
            
            return analysis
            
        except Exception as e:
            print(f"Error in AI analysis for {ticker}: {str(e)}")
            return self._fallback_analysis(ticker, stock_info, technical_signals, price_data_summary)
    
    def _prepare_context(self, ticker: str, stock_info: Dict, 
                        technical_signals: Dict, price_data_summary: Dict) -> str:
        """Prepare context string for AI"""
        context = f"""
STOCK: {ticker} ({stock_info.get('name', ticker)})
SECTOR: {stock_info.get('sector', 'Unknown')}
INDUSTRY: {stock_info.get('industry', 'Unknown')}

CURRENT PRICE: ${technical_signals.get('current_price', 0):.2f}
PRICE CHANGE (1d): {technical_signals.get('price_change_1d', 0):.2f}%
PRICE CHANGE (5d): {technical_signals.get('price_change_5d', 0):.2f}%
PRICE CHANGE (20d): {technical_signals.get('price_change_20d', 0):.2f}%

TECHNICAL INDICATORS:
- RSI: {technical_signals.get('rsi', 50):.2f} ({technical_signals.get('rsi_signal', 'neutral')})
- Trend: {technical_signals.get('direction', 'unknown')} (Strength: {technical_signals.get('strength', 0):.1f}/100)
- EMA Cross: {technical_signals.get('ema_cross', 'neutral')}
- MACD Signal: {technical_signals.get('macd_signal', 'neutral')}
- Bollinger Position: {technical_signals.get('bb_position', 'middle')}

MARKET DATA:
- 52 Week High: ${stock_info.get('52_week_high', 0):.2f}
- 52 Week Low: ${stock_info.get('52_week_low', 0):.2f}
- P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}
- Market Cap: ${stock_info.get('market_cap', 0):,.0f}
"""
        return context
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for AI"""
        return """You are an expert stock analyst specializing in identifying high-potential tech stocks and rising stocks with upside potential. 
Analyze the provided stock data and technical indicators to determine if this is a good buy opportunity RIGHT NOW.

Focus on:
1. Technical momentum and trend strength
2. Price action and recent performance
3. Relative strength compared to market
4. Risk/reward ratio
5. Entry timing

Provide a clear BUY, SELL, or WAIT recommendation with confidence level (0-100) and brief reasoning."""
    
    def _create_analysis_prompt(self, context: str) -> str:
        """Create analysis prompt"""
        return f"""Analyze this stock for a SCALPING/SHORT-TERM trading opportunity:

{context}

Based on the technical indicators and current market conditions, should I BUY this stock RIGHT NOW for short-term gains?

Provide your analysis in this format:
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
REASONING: [Brief explanation]
UPSIDE_POTENTIAL: [Low/Medium/High]
RISK_LEVEL: [Low/Medium/High]"""
    
    def _parse_ai_response(self, ai_response: str, technical_signals: Dict) -> Dict:
        """Parse AI response into structured format"""
        recommendation = "WAIT"
        confidence = 50
        reasoning = ai_response
        upside_potential = "Medium"
        risk_level = "Medium"
        
        # Extract recommendation
        if "RECOMMENDATION:" in ai_response:
            rec_line = [line for line in ai_response.split('\n') if 'RECOMMENDATION:' in line][0]
            if 'BUY' in rec_line.upper():
                recommendation = "BUY"
            elif 'SELL' in rec_line.upper():
                recommendation = "SELL"
        
        # Extract confidence
        if "CONFIDENCE:" in ai_response:
            conf_line = [line for line in ai_response.split('\n') if 'CONFIDENCE:' in line]
            if conf_line:
                try:
                    confidence = int(''.join(filter(str.isdigit, conf_line[0])))
                except:
                    pass
        
        # Extract reasoning
        if "REASONING:" in ai_response:
            reasoning_lines = []
            in_reasoning = False
            for line in ai_response.split('\n'):
                if 'REASONING:' in line:
                    in_reasoning = True
                    reasoning_lines.append(line.split('REASONING:')[1].strip())
                elif in_reasoning and line.strip() and not any(x in line.upper() for x in ['UPSIDE', 'RISK', 'CONFIDENCE']):
                    reasoning_lines.append(line.strip())
                elif in_reasoning and any(x in line.upper() for x in ['UPSIDE', 'RISK']):
                    break
            if reasoning_lines:
                reasoning = ' '.join(reasoning_lines)
        
        # Extract upside and risk
        if "UPSIDE_POTENTIAL:" in ai_response:
            upside_line = [line for line in ai_response.split('\n') if 'UPSIDE_POTENTIAL:' in line][0]
            if 'HIGH' in upside_line.upper():
                upside_potential = "High"
            elif 'LOW' in upside_line.upper():
                upside_potential = "Low"
        
        if "RISK_LEVEL:" in ai_response:
            risk_line = [line for line in ai_response.split('\n') if 'RISK_LEVEL:' in line][0]
            if 'HIGH' in risk_line.upper():
                risk_level = "High"
            elif 'LOW' in risk_line.upper():
                risk_level = "Low"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': reasoning,
            'upside_potential': upside_potential,
            'risk_level': risk_level,
            'ai_analysis': ai_response,
            'technical_score': self._calculate_technical_score(technical_signals)
        }
    
    def _calculate_technical_score(self, signals: Dict) -> float:
        """Calculate technical score (0-100) based on signals"""
        score = 50  # Base score
        
        # RSI contribution
        rsi = signals.get('rsi', 50)
        if 30 < rsi < 70:
            score += 10
        elif rsi < 30:  # Oversold - potential buy
            score += 15
        
        # Trend contribution
        direction = signals.get('direction', 'unknown')
        strength = signals.get('strength', 0)
        if direction == 'up':
            score += min(20, strength / 5)
        
        # Price change contribution
        change_5d = signals.get('price_change_5d', 0)
        if change_5d > 0:
            score += min(15, change_5d)
        
        # EMA cross contribution
        if signals.get('ema_cross') == 'bullish':
            score += 10
        
        # MACD contribution
        if signals.get('macd_signal') == 'bullish':
            score += 10
        
        return min(100, max(0, score))
    
    def _fallback_analysis(self, ticker: str, stock_info: Dict, 
                          technical_signals: Dict, price_data_summary: Dict) -> Dict:
        """Fallback analysis when AI is not available"""
        technical_score = self._calculate_technical_score(technical_signals)
        
        # Simple rule-based recommendation
        rsi = technical_signals.get('rsi', 50)
        trend = technical_signals.get('direction', 'unknown')
        change_5d = technical_signals.get('price_change_5d', 0)
        
        if technical_score >= 70 and trend == 'up' and change_5d > 0:
            recommendation = "BUY"
            confidence = min(85, technical_score)
        elif technical_score <= 30:
            recommendation = "SELL"
            confidence = 60
        else:
            recommendation = "WAIT"
            confidence = 50
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': f"Technical analysis shows {trend} trend with RSI {rsi:.1f}. Score: {technical_score:.1f}/100",
            'upside_potential': "High" if technical_score >= 70 else ("Medium" if technical_score >= 50 else "Low"),
            'risk_level': "Low" if technical_score >= 70 else ("Medium" if technical_score >= 50 else "High"),
            'technical_score': technical_score,
            'ai_analysis': "AI analysis not available - using technical indicators only"
        }

# ============================================================================
# STOCK SCREENER CLASS
# ============================================================================

class StockScreener:
    """Screen stocks based on various criteria"""
    
    def __init__(self):
        self.filters = []
    
    def add_filter(self, filter_func):
        """Add a custom filter function"""
        self.filters.append(filter_func)
    
    def screen_tech_stocks(self, stocks_data: Dict[str, pd.DataFrame], 
                          stock_infos: Dict[str, Dict]) -> List[str]:
        """Screen for tech stocks"""
        tech_tickers = []
        for ticker, info in stock_infos.items():
            sector = info.get('sector', '').lower()
            industry = info.get('industry', '').lower()
            
            tech_keywords = ['technology', 'software', 'semiconductor', 'internet', 
                           'telecommunications', 'hardware', 'cloud', 'ai', 'tech']
            
            if any(keyword in sector or keyword in industry for keyword in tech_keywords):
                tech_tickers.append(ticker)
        
        return tech_tickers
    
    def screen_rising_stocks(self, stocks_data: Dict[str, pd.DataFrame], 
                            min_change_5d: float = 2.0,
                            min_change_20d: float = 5.0) -> List[str]:
        """Screen for stocks with rising prices"""
        rising_tickers = []
        
        for ticker, df in stocks_data.items():
            if len(df) < 20:
                continue
            
            # Calculate price changes
            current_price = df['close'].iloc[-1]
            price_5d_ago = df['close'].iloc[-6] if len(df) >= 6 else current_price
            price_20d_ago = df['close'].iloc[-21] if len(df) >= 21 else current_price
            
            change_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
            change_20d = ((current_price - price_20d_ago) / price_20d_ago) * 100
            
            if change_5d >= min_change_5d and change_20d >= min_change_20d:
                rising_tickers.append(ticker)
        
        return rising_tickers
    
    def screen_momentum_stocks(self, stocks_data: Dict[str, pd.DataFrame],
                              min_momentum: float = 5.0) -> List[str]:
        """Screen for stocks with strong momentum"""
        momentum_tickers = []
        
        for ticker, df in stocks_data.items():
            if len(df) < 10:
                continue
            
            # Calculate momentum (recent price change)
            current_price = df['close'].iloc[-1]
            price_10d_ago = df['close'].iloc[-11] if len(df) >= 11 else current_price
            
            momentum = ((current_price - price_10d_ago) / price_10d_ago) * 100
            
            if momentum >= min_momentum:
                momentum_tickers.append(ticker)
        
        return momentum_tickers
    
    def screen_oversold_stocks(self, stocks_data: Dict[str, pd.DataFrame],
                              max_rsi: float = 35) -> List[str]:
        """Screen for oversold stocks (potential reversal)"""
        oversold_tickers = []
        
        for ticker, df in stocks_data.items():
            if len(df) < 20:
                continue
            
            df_with_indicators = calculate_all_indicators(df)
            if 'rsi' not in df_with_indicators.columns:
                continue
            
            current_rsi = df_with_indicators['rsi'].iloc[-1]
            
            if current_rsi <= max_rsi:
                oversold_tickers.append(ticker)
        
        return oversold_tickers
    
    def screen_breakout_stocks(self, stocks_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Screen for stocks breaking out (price above recent highs)"""
        breakout_tickers = []
        
        for ticker, df in stocks_data.items():
            if len(df) < 20:
                continue
            
            df_with_indicators = calculate_all_indicators(df)
            if 'bb_upper' not in df_with_indicators.columns:
                continue
            
            current_price = df['close'].iloc[-1]
            bb_upper = df_with_indicators['bb_upper'].iloc[-1]
            
            # Price breaking above upper Bollinger Band
            if current_price > bb_upper:
                breakout_tickers.append(ticker)
        
        return breakout_tickers
    
    def screen_by_volume(self, stocks_data: Dict[str, pd.DataFrame],
                        min_volume_ratio: float = 1.5) -> List[str]:
        """Screen for stocks with high volume (unusual activity)"""
        high_volume_tickers = []
        
        for ticker, df in stocks_data.items():
            if 'volume' not in df.columns or len(df) < 20:
                continue
            
            volume_indicators = calculate_volume_indicators(df)
            if 'volume_ratio' not in volume_indicators:
                continue
            
            current_volume_ratio = volume_indicators['volume_ratio'].iloc[-1]
            
            if current_volume_ratio >= min_volume_ratio:
                high_volume_tickers.append(ticker)
        
        return high_volume_tickers
    
    def screen_buy_opportunities(self, stocks_data: Dict[str, pd.DataFrame],
                                stock_infos: Dict[str, Dict],
                                analysis_results: Dict[str, Dict]) -> List[str]:
        """Screen for stocks with buy recommendations"""
        buy_tickers = []
        
        for ticker, analysis in analysis_results.items():
            recommendation = analysis.get('recommendation', 'WAIT')
            confidence = analysis.get('confidence', 0)
            
            if recommendation == 'BUY' and confidence >= 60:
                buy_tickers.append(ticker)
        
        return buy_tickers
    
    def comprehensive_screen(self, stocks_data: Dict[str, pd.DataFrame],
                           stock_infos: Dict[str, Dict],
                           analysis_results: Dict[str, Dict],
                           criteria: Optional[Dict] = None) -> List[str]:
        """
        Comprehensive screening with multiple criteria
        
        Args:
            stocks_data: Dictionary of ticker -> DataFrame
            stock_infos: Dictionary of ticker -> stock info
            analysis_results: Dictionary of ticker -> analysis results
            criteria: Optional criteria dict with:
                - tech_only: bool
                - min_confidence: float
                - min_upside: str (Low/Medium/High)
                - max_risk: str (Low/Medium/High)
                - rising_only: bool
        """
        if criteria is None:
            criteria = {}
        
        # Start with all tickers
        candidates = set(stocks_data.keys())
        
        # Filter by tech stocks
        if criteria.get('tech_only', False):
            tech_tickers = self.screen_tech_stocks(stocks_data, stock_infos)
            candidates = candidates.intersection(set(tech_tickers))
        
        # Filter by buy recommendations
        if 'min_confidence' in criteria:
            buy_tickers = self.screen_buy_opportunities(stocks_data, stock_infos, analysis_results)
            candidates = candidates.intersection(set(buy_tickers))
        
        # Filter by upside potential
        if 'min_upside' in criteria:
            min_upside = criteria['min_upside']
            upside_map = {'Low': 0, 'Medium': 1, 'High': 2}
            min_upside_val = upside_map.get(min_upside, 0)
            
            filtered = []
            for ticker in candidates:
                if ticker in analysis_results:
                    upside = analysis_results[ticker].get('upside_potential', 'Low')
                    if upside_map.get(upside, 0) >= min_upside_val:
                        filtered.append(ticker)
            candidates = set(filtered)
        
        # Filter by risk level
        if 'max_risk' in criteria:
            max_risk = criteria['max_risk']
            risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
            max_risk_val = risk_map.get(max_risk, 2)
            
            filtered = []
            for ticker in candidates:
                if ticker in analysis_results:
                    risk = analysis_results[ticker].get('risk_level', 'High')
                    if risk_map.get(risk, 2) <= max_risk_val:
                        filtered.append(ticker)
            candidates = set(filtered)
        
        # Filter by rising stocks
        if criteria.get('rising_only', False):
            rising_tickers = self.screen_rising_stocks(stocks_data)
            candidates = candidates.intersection(set(rising_tickers))
        
        return sorted(list(candidates))

# ============================================================================
# AGENTIC SCANNER CLASS (MAIN ORCHESTRATOR)
# ============================================================================

class AgenticStockScanner:
    """Main agentic AI system for scanning and analyzing stocks"""
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the agentic scanner
        
        Args:
            openai_api_key: OpenAI API key (optional, can use env var)
            model: AI model to use
        """
        self.ai_analyzer = StockAIAnalyzer(api_key=openai_api_key, model=model)
        self.screener = StockScreener()
        self.scan_results = {}
    
    def scan_stocks(self, tickers: Optional[List[str]] = None,
                   period: str = "3mo", interval: str = "1d",
                   focus: str = "all") -> Dict:
        """
        Scan stocks and generate buy recommendations
        
        Args:
            tickers: List of specific tickers to scan (None = scan all)
            period: Time period for data
            interval: Data interval
            focus: Focus area ("tech", "rising", "all")
        
        Returns:
            Dictionary with scan results
        """
        print(f"\n{'='*60}")
        print(f"AGENTIC AI STOCK SCANNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Fetch stock data
        print("📊 Fetching stock data...")
        if tickers:
            stocks_data = fetch_multiple_stocks(tickers, period, interval)
        else:
            if focus == "tech":
                stocks_data = get_tech_stocks(period, interval)
            elif focus == "rising":
                stocks_data = get_rising_stocks(period, interval)
            else:
                stocks_data = get_all_stocks(period, interval)
        
        print(f"✅ Fetched data for {len(stocks_data)} stocks\n")
        
        if not stocks_data:
            return {'error': 'No stock data fetched', 'recommendations': []}
        
        # Get stock information
        print("📋 Gathering stock information...")
        stock_infos = {}
        for ticker in stocks_data.keys():
            info = get_stock_info(ticker)
            stock_infos[ticker] = info
            time.sleep(0.1)  # Rate limiting
        
        print(f"✅ Collected info for {len(stock_infos)} stocks\n")
        
        # Calculate technical indicators
        print("🔧 Calculating technical indicators...")
        stocks_with_indicators = {}
        technical_signals = {}
        
        for ticker, df in stocks_data.items():
            df_indicators = calculate_all_indicators(df)
            stocks_with_indicators[ticker] = df_indicators
            signals = get_current_signals(df_indicators)
            technical_signals[ticker] = signals
        
        print(f"✅ Calculated indicators for {len(stocks_with_indicators)} stocks\n")
        
        # AI Analysis
        print("🤖 Running AI analysis...")
        analysis_results = {}
        
        for ticker in stocks_data.keys():
            print(f"  Analyzing {ticker}...", end=" ")
            
            stock_info = stock_infos.get(ticker, {})
            signals = technical_signals.get(ticker, {})
            df = stocks_with_indicators.get(ticker)
            
            price_summary = {
                'current_price': signals.get('current_price', 0),
                'price_change_1d': signals.get('price_change_1d', 0),
                'price_change_5d': signals.get('price_change_5d', 0),
                'price_change_20d': signals.get('price_change_20d', 0),
            }
            
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
            'all_analysis': analysis_results
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

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agentic AI Stock Scanner')
    parser.add_argument('--focus', choices=['tech', 'rising', 'all'], default='all',
                       help='Focus area for scanning')
    parser.add_argument('--period', default='3mo', help='Time period for data')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--limit', type=int, default=10, help='Number of recommendations to show')
    parser.add_argument('--save', action='store_true', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = AgenticStockScanner(openai_api_key=args.api_key)
    
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
