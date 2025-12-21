"""
Analysis modules - Technical and AI analysis
"""
from .technical_analyzer import (
    calculate_rsi,
    calculate_ema,
    calculate_sma,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_momentum,
    calculate_price_change,
    calculate_volume_indicators,
    analyze_trend,
    identify_demand_supply_zones,
    calculate_all_indicators,
    get_current_signals
)

from .ai_analyzer import StockAIAnalyzer

try:
    from .ml_analyzer import MLStockAnalyzer
    ML_AVAILABLE = True
except ImportError:
    MLStockAnalyzer = None
    ML_AVAILABLE = False

from .news_analyzer import get_news_sentiment_summary, fetch_stock_news, analyze_sentiment
from .agentic_supervisor import AgenticSupervisor

__all__ = [
    # Technical analysis
    'calculate_rsi',
    'calculate_ema',
    'calculate_sma',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_atr',
    'calculate_momentum',
    'calculate_price_change',
    'calculate_volume_indicators',
    'analyze_trend',
    'identify_demand_supply_zones',
    'generate_buy_recommendation',
    'calculate_all_indicators',
    'get_current_signals',
    # AI analysis
    'StockAIAnalyzer',
    # ML analysis
    'MLStockAnalyzer',
    # News & Sentiment
    'get_news_sentiment_summary',
    'fetch_stock_news',
    'analyze_sentiment',
    # Agentic Supervisor
    'AgenticSupervisor'
]
