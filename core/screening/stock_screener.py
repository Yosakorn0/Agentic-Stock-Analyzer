"""
Stock Screener - Filters and screens stocks based on criteria
"""
from typing import Dict, List, Optional
import pandas as pd

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
        from core.analysis import calculate_all_indicators
        
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
        from core.analysis import calculate_all_indicators
        
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
        from core.analysis import calculate_volume_indicators
        
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

if __name__ == "__main__":
    # Test screener
    screener = StockScreener()
    print("Stock Screener initialized")


