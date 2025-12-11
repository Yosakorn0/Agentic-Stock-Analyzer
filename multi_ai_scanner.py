"""
Multi-AI Ensemble Scanner

Uses multiple AI models in parallel to analyze stocks and generate consensus-based recommendations.
This provides higher accuracy, precision, and safety by combining multiple AI perspectives.

Supported AI Models:
- OpenAI: gpt-4o-mini, gpt-4, gpt-3.5-turbo
- Hugging Face: hf:mistralai/Mistral-7B-Instruct-v0.2, hf:meta-llama/Llama-2-7b-chat-hf, hf:AdaptLLM/finance-chat, etc.
- Technical Analysis Fallback (when AI unavailable)

Usage:
    python multi_ai_scanner.py
    python multi_ai_scanner.py --models gpt-4o-mini,gpt-4,hf:mistralai/Mistral-7B-Instruct-v0.2
    python multi_ai_scanner.py --preset finance-wide
    python multi_ai_scanner.py --min-consensus 75 --min-score 70
"""

import sys
import os
from typing import List, Optional, Dict
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import scanner components
try:
    from stock_scanner import AgenticStockScanner, StockAIAnalyzer, fetch_multiple_stocks, get_all_stocks, get_tech_stocks, get_rising_stocks
    from stock_scanner import calculate_all_indicators, get_current_signals, get_stock_info
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scanners.agentic_scanner import AgenticStockScanner
    from core.analysis.ai_analyzer import StockAIAnalyzer
    from core.data.stock_fetcher import fetch_multiple_stocks, get_all_stocks, get_tech_stocks, get_rising_stocks
    from core.analysis.technical_analyzer import calculate_all_indicators, get_current_signals
    from core.data.stock_fetcher import get_stock_info

# Import Hugging Face analyzer
try:
    from core.analysis.huggingface_analyzer import HuggingFaceAnalyzer, RECOMMENDED_MODELS
    HUGGINGFACE_ANALYZER_AVAILABLE = True
except ImportError:
    HUGGINGFACE_ANALYZER_AVAILABLE = False
    print("⚠️ Warning: Hugging Face analyzer not available. Install with: pip install transformers torch")

# Preset model bundles
PRESET_MODELS = {
    # Balanced, cost-aware
    "diversified-6": [
        "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo",
        "hf:mistral-7b", "hf:llama-2-7b", "hf:zephyr-7b"
    ],
    # Finance-heavy (LLMs tuned for finance)
    "finance-wide": [
        "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo",
        "hf:finance-chat",
        "hf:llama-open-finance-8b",
        "hf:qwen-open-finance-r-8b",
        "hf:fin-o1-14b",
        "hf:mistral-7b",
        "hf:llama-2-7b",
        "hf:zephyr-7b"
    ],
    # OpenAI only (fast setup)
    "openai-only": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
}


class MultiAIAnalyzer:
    """
    Analyzes stocks using multiple AI models in parallel and creates consensus
    """
    
    def __init__(self, models: List[str] = None, api_key: Optional[str] = None, use_gpu: bool = False):
        """
        Initialize multi-AI analyzer
        
        Args:
            models: List of AI models to use
                   OpenAI: 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo'
                   Hugging Face: 'hf:model-name' or 'hf:mistralai/Mistral-7B-Instruct-v0.2'
                   Shortcuts: 'hf:mistral-7b', 'hf:llama-2-7b', etc.
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            use_gpu: Whether to use GPU for Hugging Face models
        """
        if models is None:
            models = ['gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo']
        
        self.models = models
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_gpu = use_gpu
        self.analyzers = {}
        
        # Initialize analyzers for each model
        for model in models:
            try:
                if model.startswith('hf:') or '/' in model:
                    # Hugging Face model
                    if not HUGGINGFACE_ANALYZER_AVAILABLE:
                        print(f"⚠️ Warning: Hugging Face analyzer not available, skipping {model}")
                        continue
                    
                    # Extract model name (remove 'hf:' prefix if present)
                    hf_model_name = model.replace('hf:', '', 1)
                    
                    # Check if it's a shortcut
                    if hf_model_name in RECOMMENDED_MODELS:
                        hf_model_name = RECOMMENDED_MODELS[hf_model_name]
                        print(f"📦 Using recommended model: {hf_model_name}")
                    
                    self.analyzers[model] = HuggingFaceAnalyzer(
                        model_name=hf_model_name,
                        use_gpu=self.use_gpu
                    )
                    print(f"✅ Initialized Hugging Face model: {model} ({hf_model_name})")
                else:
                    # OpenAI model
                    self.analyzers[model] = StockAIAnalyzer(api_key=self.api_key, model=model)
                    print(f"✅ Initialized OpenAI model: {model}")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize {model}: {e}")
    
    def analyze_stock_multi_ai(
        self,
        ticker: str,
        stock_info: Dict,
        technical_signals: Dict,
        price_data_summary: Dict
    ) -> Dict:
        """
        Analyze a stock using multiple AI models in parallel
        
        Returns:
            Dictionary with consensus analysis and individual model results
        """
        results = {}
        
        # Analyze with each AI model in parallel
        with ThreadPoolExecutor(max_workers=len(self.analyzers)) as executor:
            futures = {
                executor.submit(
                    analyzer.analyze_stock,
                    ticker, stock_info, technical_signals, price_data_summary
                ): model_name
                for model_name, analyzer in self.analyzers.items()
            }
            
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results[model_name] = result
                except Exception as e:
                    print(f"⚠️ Error with {model_name}: {e}")
                    results[model_name] = None
        
        # Calculate consensus (with entry price recommendations)
        consensus = self._calculate_consensus(results, technical_signals, stock_info)
        
        return {
            'individual_results': results,
            'consensus': consensus,
            'ticker': ticker
        }
    
    def _calculate_consensus(self, individual_results: Dict, technical_signals: Dict, stock_info: Dict = None) -> Dict:
        """
        Calculate consensus from multiple AI model results and add entry price recommendations
        
        Args:
            individual_results: Dict mapping model_name -> analysis_result
            technical_signals: Technical signals for entry price calculation
            stock_info: Stock information (optional)
        
        Returns:
            Consensus analysis result with entry price recommendations
        """
        valid_results = {k: v for k, v in individual_results.items() if v is not None}
        
        if not valid_results:
            # All models failed, use technical fallback
            return self._technical_fallback(technical_signals)
        
        # Extract recommendations and confidences
        recommendations = []
        confidences = []
        reasonings = []
        upside_potentials = []
        risk_levels = []
        
        for model_name, result in valid_results.items():
            if result:
                recommendations.append(result.get('recommendation', 'WAIT'))
                confidences.append(result.get('confidence', 50))
                reasonings.append(result.get('reasoning', ''))
                upside_potentials.append(result.get('upside_potential', 'Medium'))
                risk_levels.append(result.get('risk_level', 'Medium'))
        
        # Calculate consensus recommendation (majority vote)
        buy_count = recommendations.count('BUY')
        consider_buy_count = recommendations.count('CONSIDER BUY')
        wait_count = recommendations.count('WAIT')
        sell_count = recommendations.count('SELL')
        
        total_votes = len(recommendations)
        
        if buy_count > total_votes / 2:
            consensus_recommendation = 'BUY'
        elif (buy_count + consider_buy_count) > total_votes / 2:
            consensus_recommendation = 'CONSIDER BUY'
        elif sell_count > total_votes / 2:
            consensus_recommendation = 'SELL'
        else:
            consensus_recommendation = 'WAIT'
        
        # Calculate average confidence (weighted by recommendation strength)
        weighted_confidences = []
        for rec, conf in zip(recommendations, confidences):
            if rec == 'BUY':
                weight = 1.2  # Boost BUY confidence
            elif rec == 'CONSIDER BUY':
                weight = 1.0
            elif rec == 'WAIT':
                weight = 0.8  # Reduce WAIT confidence
            else:  # SELL
                weight = 0.9
            weighted_confidences.append(conf * weight)
        
        avg_confidence = sum(weighted_confidences) / len(weighted_confidences) if weighted_confidences else 50
        avg_confidence = min(100, max(0, avg_confidence))  # Clamp 0-100
        
        # Consensus confidence: higher if models agree, lower if they disagree
        confidence_variance = self._calculate_variance(confidences)
        agreement_factor = 1.0 - (confidence_variance / 2500.0)  # Normalize variance
        consensus_confidence = avg_confidence * agreement_factor
        
        # Majority agreement percentage
        recommendation_counts = {
            'BUY': buy_count,
            'CONSIDER BUY': consider_buy_count,
            'WAIT': wait_count,
            'SELL': sell_count
        }
        max_count = max(recommendation_counts.values())
        agreement_pct = (max_count / total_votes) * 100 if total_votes > 0 else 0
        
        # Consensus upside and risk (most common)
        upside_counts = {}
        risk_counts = {}
        for up, risk in zip(upside_potentials, risk_levels):
            upside_counts[up] = upside_counts.get(up, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        consensus_upside = max(upside_counts, key=upside_counts.get) if upside_counts else 'Medium'
        consensus_risk = max(risk_counts, key=risk_counts.get) if risk_counts else 'Medium'
        
        # Combine reasonings
        consensus_reasoning = self._combine_reasonings(reasonings, recommendations, consensus_recommendation)
        
        # Calculate technical score from signals
        technical_score = self._calculate_technical_score(technical_signals)
        
        # Calculate entry price recommendations if BUY or CONSIDER BUY
        entry_info = self._calculate_entry_price(
            consensus_recommendation, 
            technical_signals,
            technical_score
        )
        
        consensus_result = {
            'recommendation': consensus_recommendation,
            'confidence': round(consensus_confidence, 1),
            'reasoning': consensus_reasoning,
            'upside_potential': consensus_upside,
            'risk_level': consensus_risk,
            'technical_score': technical_score,
            'agreement_percentage': round(agreement_pct, 1),
            'models_analyzed': len(valid_results),
            'model_breakdown': {
                'BUY': buy_count,
                'CONSIDER BUY': consider_buy_count,
                'WAIT': wait_count,
                'SELL': sell_count
            },
            'confidence_range': {
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0,
                'avg': round(sum(confidences) / len(confidences), 1) if confidences else 0
            }
        }
        
        # Add entry price info if available
        consensus_result.update(entry_info)
        
        return consensus_result
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _combine_reasonings(self, reasonings: List[str], recommendations: List[str], consensus: str) -> str:
        """Combine multiple AI reasonings into consensus reasoning"""
        if not reasonings:
            return "No AI analysis available"
        
        # Count how many models agree with consensus
        agreeing_indices = [i for i, rec in enumerate(recommendations) if rec == consensus]
        
        if agreeing_indices:
            # Use reasoning from models that agree
            agreeing_reasonings = [reasonings[i] for i in agreeing_indices]
            combined = " | ".join(agreeing_reasonings[:2])  # Use first 2 agreeing models
            if len(agreeing_indices) > 2:
                combined += f" ({len(agreeing_indices)} models agree)"
            return combined
        else:
            # Mixed signals
            return f"Mixed signals: {len(reasonings)} models analyzed | {reasonings[0]}"
    
    def _calculate_technical_score(self, signals: Dict) -> float:
        """Calculate technical score (fallback method)"""
        score = 50  # Base score
        
        rsi = signals.get('rsi', 50)
        if 30 < rsi < 70:
            score += 10
        elif rsi < 30:
            score += 15
        
        direction = signals.get('direction', 'unknown')
        strength = signals.get('strength', 0)
        if direction == 'up':
            score += min(20, strength / 5)
        
        change_5d = signals.get('price_change_5d', 0)
        if change_5d > 0:
            score += min(15, change_5d)
        
        if signals.get('ema_cross') == 'bullish':
            score += 10
        
        if signals.get('macd_signal') == 'bullish':
            score += 10
        
        return min(100, max(0, score))
    
    def _calculate_entry_price(self, recommendation: str, technical_signals: Dict, technical_score: float) -> Dict:
        """
        Calculate entry price, stop-loss, and take-profit based on technical signals
        
        Args:
            recommendation: Consensus recommendation (BUY, CONSIDER BUY, etc.)
            technical_signals: Technical analysis signals
            technical_score: Technical analysis score
        
        Returns:
            Dictionary with entry price recommendations
        """
        current_price = technical_signals.get('current_price', 0)
        rsi = technical_signals.get('rsi', 50)
        trend_direction = technical_signals.get('direction', 'unknown')
        nearest_demand = technical_signals.get('nearest_demand')
        nearest_supply = technical_signals.get('nearest_supply')
        distance_to_demand = technical_signals.get('distance_to_demand_pct', 100)
        
        suggested_entry_price = None
        entry_price_reason = None
        stop_loss = None
        take_profit = None
        risk_reward_ratio = None
        
        if recommendation in ['BUY', 'CONSIDER BUY']:
            # Strategy 1: If near demand zone, buy at or slightly above demand zone
            if nearest_demand and distance_to_demand < 5:
                suggested_entry_price = nearest_demand * 1.01  # 1% above demand zone
                entry_price_reason = f"Enter near demand zone (${nearest_demand:,.2f})"
            # Strategy 2: If oversold, buy at current price or slightly below
            elif rsi < 35:
                suggested_entry_price = current_price * 0.995  # 0.5% below current
                entry_price_reason = "Enter on oversold bounce"
            # Strategy 3: If in uptrend, buy on pullback to support (EMA 21 or demand zone)
            elif trend_direction == 'up' and nearest_demand:
                suggested_entry_price = max(nearest_demand, current_price * 0.98)
                entry_price_reason = "Buy on pullback in uptrend"
            # Strategy 4: Default - current price with small discount
            else:
                suggested_entry_price = current_price * 0.99
                entry_price_reason = "Enter at slight discount to current price"
            
            # Calculate stop loss (below nearest demand or 3% below entry)
            if nearest_demand:
                stop_loss = nearest_demand * 0.97  # 3% below demand zone
            else:
                stop_loss = suggested_entry_price * 0.97  # 3% below entry
            
            # Calculate take profit (near supply zone or 5-10% above entry)
            if nearest_supply and nearest_supply > suggested_entry_price:
                profit_pct = ((nearest_supply - suggested_entry_price) / suggested_entry_price) * 100
                if profit_pct < 15:  # Only use if reasonable
                    take_profit = nearest_supply * 0.99  # Just below supply
                else:
                    take_profit = suggested_entry_price * 1.08  # 8% profit target
            else:
                take_profit = suggested_entry_price * 1.08  # 8% profit target
            
            # Calculate risk/reward ratio
            risk = suggested_entry_price - stop_loss
            reward = take_profit - suggested_entry_price
            if risk > 0:
                risk_reward_ratio = reward / risk
            else:
                risk_reward_ratio = None
        else:
            # For WAIT/WATCH, suggest waiting for better entry
            if nearest_demand:
                suggested_entry_price = nearest_demand * 1.02
                entry_price_reason = f"Wait for pullback to demand zone (${nearest_demand:,.2f})"
            else:
                suggested_entry_price = current_price * 0.95
                entry_price_reason = "Wait for 5% pullback"
        
        return {
            'suggested_entry_price': float(suggested_entry_price) if suggested_entry_price else None,
            'entry_price_reason': entry_price_reason,
            'stop_loss': float(stop_loss) if stop_loss else None,
            'take_profit': float(take_profit) if take_profit else None,
            'risk_reward_ratio': float(risk_reward_ratio) if risk_reward_ratio else None
        }
    
    def _technical_fallback(self, signals: Dict) -> Dict:
        """Fallback to technical analysis when all AI models fail"""
        technical_score = self._calculate_technical_score(signals)
        rsi = signals.get('rsi', 50)
        trend = signals.get('direction', 'unknown')
        
        if technical_score >= 70:
            recommendation = "BUY"
            confidence = min(75, technical_score)
        elif technical_score <= 30:
            recommendation = "SELL"
            confidence = 60
        else:
            recommendation = "WAIT"
            confidence = 50
        
        # Calculate entry price for fallback
        entry_info = self._calculate_entry_price(recommendation, signals, technical_score)
        
        result = {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': f"Technical analysis only (AI unavailable): {trend} trend, RSI {rsi:.1f}, Score {technical_score:.1f}",
            'upside_potential': "High" if technical_score >= 70 else ("Medium" if technical_score >= 50 else "Low"),
            'risk_level': "Low" if technical_score >= 70 else ("Medium" if technical_score >= 50 else "High"),
            'technical_score': technical_score,
            'agreement_percentage': 0,
            'models_analyzed': 0,
            'model_breakdown': {},
            'confidence_range': {'min': confidence, 'max': confidence, 'avg': confidence}
        }
        
        result.update(entry_info)
        return result


def scan_multi_ai(
    min_score: int = 60,
    min_consensus: float = 60.0,
    focus: str = "all",
    tickers: Optional[List[str]] = None,
    period: str = "3mo",
    interval: str = "1d",
    models: List[str] = None,
    use_gpu: bool = False
) -> List[Dict]:
    """
    Scan stocks using multiple AI models in parallel
    
    Args:
        min_score: Minimum consensus confidence score (default: 60)
        min_consensus: Minimum agreement percentage between models (default: 60%)
        focus: Focus area - "all", "tech", or "rising"
        tickers: Custom list of tickers to scan
        period: Time period for data
        interval: Data interval
        models: List of AI models to use (OpenAI or Hugging Face)
        use_gpu: Whether to use GPU for Hugging Face models
    
    Returns:
        List of high-confidence buy recommendations with multi-AI consensus
    """
    print("=" * 80)
    print(f"🤖 MULTI-AI ENSEMBLE SCANNER")
    print("=" * 80)
    model_list = models or ['gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo']
    print(f"AI Models: {', '.join(model_list)}")
    if use_gpu:
        print(f"GPU: Enabled for Hugging Face models")
    print(f"Minimum Score: {min_score}+ | Minimum Consensus: {min_consensus}%")
    print(f"Focus: {focus.upper()} | Period: {period} | Interval: {interval}")
    print("=" * 80)
    print()
    
    # Initialize multi-AI analyzer
    multi_ai = MultiAIAnalyzer(models=models, use_gpu=use_gpu)
    
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
        return []
    
    # Get stock info and calculate indicators
    print("🔧 Calculating technical indicators...")
    stock_infos = {}
    stocks_with_indicators = {}
    technical_signals = {}
    
    for ticker, df in stocks_data.items():
        stock_infos[ticker] = get_stock_info(ticker)
        df_indicators = calculate_all_indicators(df)
        stocks_with_indicators[ticker] = df_indicators
        technical_signals[ticker] = get_current_signals(df_indicators)
        time.sleep(0.1)  # Rate limiting
    
    print(f"✅ Prepared {len(stocks_with_indicators)} stocks for AI analysis\n")
    
    # Analyze each stock with multiple AIs
    print(f"🤖 Analyzing with {len(multi_ai.analyzers)} AI models in parallel...")
    high_confidence_buys = []
    
    for i, (ticker, df) in enumerate(stocks_with_indicators.items(), 1):
        print(f"  [{i}/{len(stocks_with_indicators)}] Analyzing {ticker}...", end=" ")
        
        stock_info = stock_infos.get(ticker, {})
        signals = technical_signals.get(ticker, {})
        price_summary = {
            'current_price': signals.get('current_price', 0),
            'price_change_1d': signals.get('price_change_1d', 0),
            'price_change_5d': signals.get('price_change_5d', 0),
            'price_change_20d': signals.get('price_change_20d', 0),
        }
        
        # Multi-AI analysis (pass stock_info for entry price calculation)
        result = multi_ai.analyze_stock_multi_ai(
            ticker=ticker,
            stock_info=stock_info,
            technical_signals=signals,
            price_data_summary=price_summary
        )
        
        consensus = result['consensus']
        
        # Filter for high-confidence buys with good consensus
        recommendation = consensus.get('recommendation', 'WAIT')
        confidence = consensus.get('confidence', 0)
        agreement = consensus.get('agreement_percentage', 0)
        
        # Choose icon
        if recommendation == 'BUY':
            icon = '🟢'
        elif recommendation == 'CONSIDER BUY':
            icon = '🟡'
        elif recommendation == 'WATCH':
            icon = '🟠'
        else:
            icon = '🔴'
        
        print(f"{icon} {recommendation} ({confidence:.1f}%, {agreement:.0f}% consensus)")
        
        # Only include if meets criteria
        if recommendation in ['BUY', 'CONSIDER BUY'] and confidence >= min_score and agreement >= min_consensus:
            result['stock_info'] = stock_info
            result['technical_signals'] = signals
            high_confidence_buys.append(result)
        
        time.sleep(0.2)  # Rate limiting between stocks
    
    print(f"\n✅ Found {len(high_confidence_buys)} stocks meeting criteria\n")
    
    return high_confidence_buys


def display_multi_ai_results(results: List[Dict], min_score: int, min_consensus: float):
    """Display multi-AI consensus results"""
    if not results:
        print("\n" + "=" * 80)
        print(f"❌ NO HIGH CONFIDENCE BUYS FOUND")
        print("=" * 80)
        print(f"No stocks found with:")
        print(f"  - Consensus score >= {min_score}")
        print(f"  - Agreement >= {min_consensus}%")
        return
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x['consensus'].get('confidence', 0), reverse=True)
    
    print("\n" + "=" * 80)
    print(f"✅ FOUND {len(results)} HIGH CONFIDENCE BUYS (Multi-AI Consensus)")
    print("=" * 80)
    print()
    
    # Summary table
    print(f"{'Rank':<6} {'Ticker':<8} {'Score':<8} {'Agreement':<12} {'Models':<8} {'Recommendation':<15}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        ticker = result['ticker']
        consensus = result['consensus']
        confidence = consensus.get('confidence', 0)
        agreement = consensus.get('agreement_percentage', 0)
        models_count = consensus.get('models_analyzed', 0)
        recommendation = consensus.get('recommendation', 'WAIT')
        
        if confidence >= 80:
            icon = "🟢"
        elif confidence >= 70:
            icon = "🟡"
        else:
            icon = "🟠"
        
        print(f"{i:<6} {icon} {ticker:<6} {confidence:>5.1f}%    {agreement:>5.1f}%       {models_count:<8} {recommendation:<15}")
    
    print("\n" + "=" * 80)
    print("📋 DETAILED MULTI-AI CONSENSUS ANALYSIS")
    print("=" * 80)
    print()
    
    # Detailed information
    for i, result in enumerate(results, 1):
        ticker = result['ticker']
        stock_info = result.get('stock_info', {})
        signals = result.get('technical_signals', {})
        consensus = result['consensus']
        individual = result.get('individual_results', {})
        
        name = stock_info.get('name', 'N/A')
        sector = stock_info.get('sector', 'Unknown')
        confidence = consensus.get('confidence', 0)
        recommendation = consensus.get('recommendation', 'WAIT')
        agreement = consensus.get('agreement_percentage', 0)
        reasoning = consensus.get('reasoning', 'N/A')
        upside = consensus.get('upside_potential', 'Medium')
        risk = consensus.get('risk_level', 'Medium')
        technical_score = consensus.get('technical_score', 0)
        model_breakdown = consensus.get('model_breakdown', {})
        confidence_range = consensus.get('confidence_range', {})
        models_analyzed = consensus.get('models_analyzed', 0)
        
        price = signals.get('current_price', 0)
        change_5d = signals.get('price_change_5d', 0)
        trend = signals.get('direction', 'unknown')
        rsi = signals.get('rsi', 0)
        
        if confidence >= 80:
            icon = "🟢"
            quality = "EXCELLENT"
        elif confidence >= 70:
            icon = "🟡"
            quality = "VERY GOOD"
        else:
            icon = "🟠"
            quality = "GOOD"
        
        print(f"{i}. {icon} {ticker} - {name}")
        print(f"   Sector: {sector}")
        print(f"   📊 CONSENSUS: {recommendation} | Score: {confidence:.1f}% ({quality})")
        print(f"   🤝 Agreement: {agreement:.1f}% | Models Analyzed: {models_analyzed}")
        print(f"   💰 Price: ${price:,.2f} | Change 5d: {change_5d:+.2f}%")
        print(f"   📈 Trend: {trend.upper()} | RSI: {rsi:.1f} | Technical Score: {technical_score:.1f}")
        print(f"   ⚠️  Risk: {risk} | 📈 Potential: {upside}")
        print(f"   📝 Reasoning: {reasoning[:250]}..." if len(reasoning) > 250 else f"   📝 Reasoning: {reasoning}")
        
        # Display entry price recommendations if BUY or CONSIDER BUY
        if recommendation in ['BUY', 'CONSIDER BUY']:
            entry_price = consensus.get('suggested_entry_price')
            stop_loss = consensus.get('stop_loss')
            take_profit = consensus.get('take_profit')
            risk_reward = consensus.get('risk_reward_ratio')
            entry_reason = consensus.get('entry_price_reason')
            
            if entry_price:
                print()
                print(f"   💰 ENTRY PRICE RECOMMENDATION:")
                print(f"      Entry Price: ${entry_price:,.2f}")
                if entry_reason:
                    print(f"      Strategy: {entry_reason}")
                
                if stop_loss:
                    stop_loss_pct = ((current_price - stop_loss) / current_price) * 100 if current_price > 0 else 0
                    print(f"      Stop Loss: ${stop_loss:,.2f} ({abs(stop_loss_pct):.2f}% below current)")
                
                if take_profit:
                    take_profit_pct = ((take_profit - current_price) / current_price) * 100 if current_price > 0 else 0
                    print(f"      Take Profit: ${take_profit:,.2f} ({take_profit_pct:.2f}% above current)")
                
                if risk_reward:
                    print(f"      Risk/Reward Ratio: 1:{risk_reward:.2f}")
                    if risk_reward >= 2.0:
                        print(f"      ✅ Excellent R:R ratio for trading!")
                    elif risk_reward >= 1.5:
                        print(f"      ✅ Good R:R ratio")
                    else:
                        print(f"      ⚠️  Low R:R ratio - consider tighter stop or higher target")
                
                # Show price difference
                if entry_price < current_price:
                    discount = ((current_price - entry_price) / current_price) * 100
                    print(f"      💡 Entry is ${current_price - entry_price:.2f} ({discount:.2f}%) below current - wait for pullback")
                elif entry_price > current_price:
                    premium = ((entry_price - current_price) / current_price) * 100
                    print(f"      ⚠️  Entry is ${entry_price - current_price:.2f} ({premium:.2f}%) above current - may need to wait")
                else:
                    print(f"      ✅ Entry at current price is reasonable")
        
        print()
        
        # Show individual model breakdown
        print(f"   🤖 INDIVIDUAL AI MODEL RESULTS:")
        for model_name, model_result in individual.items():
            if model_result:
                model_rec = model_result.get('recommendation', 'N/A')
                model_conf = model_result.get('confidence', 0)
                model_icon = "🟢" if model_rec == "BUY" else ("🟡" if model_rec == "CONSIDER BUY" else "🔴")
                print(f"      {model_icon} {model_name}: {model_rec} ({model_conf}%)")
        print()
        
        # Model agreement breakdown
        if model_breakdown:
            print(f"   📊 VOTE BREAKDOWN: ", end="")
            breakdown_str = ", ".join([f"{k}: {v}" for k, v in model_breakdown.items() if v > 0])
            print(breakdown_str)
            print(f"   📈 Confidence Range: {confidence_range.get('min', 0):.1f}% - {confidence_range.get('max', 0):.1f}% (avg: {confidence_range.get('avg', 0):.1f}%)")
        print()
    
    print("=" * 80)
    print("💡 MULTI-AI ENSEMBLE BENEFITS")
    print("=" * 80)
    print("• ✅ Higher accuracy: Multiple AI perspectives reduce errors")
    print("• ✅ Better precision: Consensus filtering eliminates outliers")
    print("• ✅ Risk reduction: Only trade when multiple AIs agree")
    print("• ✅ Confidence scoring: Agreement percentage shows reliability")
    print("• ✅ Model diversity: Different models catch different patterns")
    print()
    print("🎯 TRADING RECOMMENDATIONS:")
    print("• Only trade when agreement >= 70% (strong consensus)")
    print("• Higher scores (80+) with high agreement are safest")
    print("• Monitor model breakdowns for potential risks")
    print("• Use stop-losses even with high confidence")
    print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Multi-AI Ensemble Scanner for safer, more accurate stock analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default OpenAI models
  python multi_ai_scanner.py
  
  # Mix OpenAI and Hugging Face models
  python multi_ai_scanner.py --models gpt-4o-mini,hf:mistral-7b,hf:llama-2-7b
  
  # Use preset bundles
  python multi_ai_scanner.py --preset diversified-6
  python multi_ai_scanner.py --preset finance-wide --use-gpu
  
  # Use specific Hugging Face model
  python multi_ai_scanner.py --models hf:mistralai/Mistral-7B-Instruct-v0.2
  
  # Use GPU for Hugging Face models (faster)
  python multi_ai_scanner.py --models hf:mistral-7b --use-gpu
  
  # Higher thresholds for maximum safety
  python multi_ai_scanner.py --min-score 75 --min-consensus 75
  
  # Focus on tech stocks
  python multi_ai_scanner.py --focus tech --min-consensus 70
  
  # Recommended Hugging Face model shortcuts:
  # hf:mistral-7b, hf:llama-2-7b, hf:zephyr-7b, hf:gemma-7b, hf:phi-2, hf:tinyllama
        """
    )
    
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of AI models. OpenAI: gpt-4o-mini, gpt-4, gpt-3.5-turbo. Hugging Face: hf:model-name'
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=list(PRESET_MODELS.keys()),
        help='Use a preset bundle of models (e.g., diversified-6, finance-wide, openai-only)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for Hugging Face models (requires CUDA)'
    )
    
    parser.add_argument(
        '--min-score',
        type=int,
        default=60,
        help='Minimum consensus confidence score (default: 60)'
    )
    
    parser.add_argument(
        '--min-consensus',
        type=float,
        default=60.0,
        help='Minimum agreement percentage between models (default: 60.0)'
    )
    
    parser.add_argument(
        '--focus',
        type=str,
        choices=['all', 'tech', 'rising'],
        default='all',
        help='Focus area (default: all)'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated list of tickers to scan'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='3mo',
        help='Time period for data (default: 3mo)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        help='Data interval (default: 1d)'
    )
    
    args = parser.parse_args()
    
    # Parse models
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(',') if m.strip()]
    elif args.preset:
        models = PRESET_MODELS.get(args.preset)
    
    # Parse tickers
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    
    # Run scanner
    try:
        results = scan_multi_ai(
            min_score=args.min_score,
            min_consensus=args.min_consensus,
            focus=args.focus,
            tickers=tickers,
            period=args.period,
            interval=args.interval,
            models=models,
            use_gpu=args.use_gpu
        )
        
        display_multi_ai_results(results, args.min_score, args.min_consensus)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Scan interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

