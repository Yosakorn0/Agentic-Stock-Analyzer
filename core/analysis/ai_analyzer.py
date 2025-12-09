"""
AI Model Integration - Uses AI to analyze stocks and generate buy recommendations
"""
import os
from typing import Dict, List, Optional
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")

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
    
    def batch_analyze(self, stocks_data: List[Dict]) -> List[Dict]:
        """
        Analyze multiple stocks
        
        Args:
            stocks_data: List of dicts with ticker, stock_info, technical_signals, price_data_summary
        
        Returns:
            List of analysis results
        """
        results = []
        for stock_data in stocks_data:
            analysis = self.analyze_stock(**stock_data)
            results.append({
                'ticker': stock_data['ticker'],
                **analysis
            })
        return results

if __name__ == "__main__":
    # Test the AI analyzer
    analyzer = StockAIAnalyzer()
    
    # Sample data
    test_data = {
        'ticker': 'AAPL',
        'stock_info': {'name': 'Apple Inc.', 'sector': 'Technology'},
        'technical_signals': {
            'rsi': 45,
            'direction': 'up',
            'strength': 75,
            'current_price': 180.50,
            'price_change_5d': 3.5,
            'ema_cross': 'bullish',
            'macd_signal': 'bullish'
        },
        'price_data_summary': {}
    }
    
    result = analyzer.analyze_stock(**test_data)
    print("AI Analysis Result:")
    print(json.dumps(result, indent=2))


