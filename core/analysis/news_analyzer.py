"""
News and Sentiment Analysis for Stocks
Fetches recent news and analyzes sentiment
"""
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


def fetch_stock_news(ticker: str, days: int = 7, api_key: Optional[str] = None, timeout: int = 30) -> List[Dict]:
    """
    Fetch recent news articles for a stock
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to look back
        api_key: NewsAPI key (or use NEWSAPI_KEY env var)
        timeout: Request timeout in seconds
    
    Returns:
        List of news articles with title, description, sentiment
    """
    api_key = api_key or os.getenv('NEWSAPI_KEY')
    
    if not NEWSAPI_AVAILABLE or not api_key:
        # Fallback: Try yfinance news
        return _fetch_yfinance_news(ticker, days)
    
    try:
        newsapi = NewsApiClient(api_key=api_key)
        
        # Search for news about the stock
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Use timeout wrapper
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                newsapi.get_everything,
                q=ticker,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=20
            )
            articles = future.result(timeout=timeout)
        
        return articles.get('articles', [])
    except Exception as e:
        print(f"NewsAPI error: {str(e)[:100]}")
        return _fetch_yfinance_news(ticker, days)


def _fetch_yfinance_news(ticker: str, days: int) -> List[Dict]:
    """Fallback to yfinance news"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_news = []
        
        for article in news[:20]:  # Limit to 20
            pub_time = article.get('providerPublishTime', 0)
            if pub_time:
                pub_date = datetime.fromtimestamp(pub_time)
                if pub_date >= cutoff_date:
                    filtered_news.append({
                        'title': article.get('title', ''),
                        'description': article.get('summary', ''),
                        'url': article.get('link', ''),
                        'publishedAt': pub_date.isoformat(),
                        'source': {'name': article.get('publisher', 'Unknown')}
                    })
        
        return filtered_news
    except Exception as e:
        return []


def analyze_sentiment(text: str) -> Dict:
    """
    Analyze sentiment of text
    
    Returns:
        Dict with polarity (-1 to 1), subjectivity (0 to 1), and label
    """
    if not TEXTBLOB_AVAILABLE:
        # Simple keyword-based fallback
        return _simple_sentiment(text)
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'label': label
        }
    except Exception as e:
        return _simple_sentiment(text)


def _simple_sentiment(text: str) -> Dict:
    """Simple keyword-based sentiment analysis"""
    positive_words = ['up', 'rise', 'gain', 'surge', 'rally', 'bullish', 'beat', 'exceed', 'growth', 'profit', 'strong', 'positive']
    negative_words = ['down', 'fall', 'drop', 'decline', 'crash', 'bearish', 'miss', 'loss', 'warn', 'risk', 'weak', 'negative']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        polarity = 0.3
        label = 'positive'
    elif negative_count > positive_count:
        polarity = -0.3
        label = 'negative'
    else:
        polarity = 0.0
        label = 'neutral'
    
    return {
        'polarity': polarity,
        'subjectivity': 0.5,
        'label': label
    }


def get_news_sentiment_summary(ticker: str, days: int = 7, timeout: int = 30) -> Dict:
    """
    Get comprehensive news sentiment summary
    
    Returns:
        Dict with sentiment scores, news count, key themes
    """
    articles = fetch_stock_news(ticker, days, timeout=timeout)
    
    if not articles:
        return {
            'news_count': 0,
            'avg_sentiment': 0.0,
            'sentiment_label': 'neutral',
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'impact_score': 0.0,
            'recent_news': []
        }
    
    sentiments = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title} {description}"
        
        sentiment = analyze_sentiment(text)
        sentiments.append(sentiment['polarity'])
        
        if sentiment['label'] == 'positive':
            positive_count += 1
        elif sentiment['label'] == 'negative':
            negative_count += 1
        else:
            neutral_count += 1
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    
    # Impact score: combines sentiment strength and news volume
    impact_score = abs(avg_sentiment) * min(len(articles) / 10, 1.0) * 100
    
    if avg_sentiment > 0.1:
        sentiment_label = 'positive'
    elif avg_sentiment < -0.1:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    
    return {
        'news_count': len(articles),
        'avg_sentiment': avg_sentiment,
        'sentiment_label': sentiment_label,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'impact_score': impact_score,
        'recent_news': articles[:5]  # Top 5 most recent
    }

