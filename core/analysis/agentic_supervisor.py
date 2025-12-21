"""
Agentic Feedback Layer - AI Supervisor
Evaluates other AI predictions and provides meta-analysis
"""
import os
from typing import Dict, List, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AgenticSupervisor:
    """
    AI Supervisor that evaluates other AI predictions and provides meta-analysis
    Acts as a "meta-AI" that reviews all predictions and identifies contradictions
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the AI supervisor
        
        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Model to use for supervision (default: gpt-4o-mini)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {str(e)[:100]}")
    
    def evaluate_ai_predictions(
        self,
        ticker: str,
        individual_results: Dict,
        consensus: Dict,
        technical_signals: Dict,
        news_sentiment: Dict,
        fundamentals: Dict,
        stock_info: Dict
    ) -> Dict:
        """
        Evaluate all AI predictions and provide meta-analysis
        
        Args:
            ticker: Stock ticker
            individual_results: Dict of individual AI model results
            consensus: Consensus result from multi-AI
            technical_signals: Technical analysis signals
            news_sentiment: News sentiment analysis
            fundamentals: Fundamental data
            stock_info: Basic stock information
        
        Returns:
            Dict with supervisor evaluation, contradictions, and final recommendation
        """
        if not self.client:
            # Fallback: Rule-based evaluation
            return self._rule_based_evaluation(
                individual_results, consensus, technical_signals, news_sentiment, fundamentals
            )
        
        try:
            # Prepare context for supervisor
            context = self._prepare_supervisor_context(
                ticker, individual_results, consensus, technical_signals,
                news_sentiment, fundamentals, stock_info
            )
            
            # Create supervisor prompt
            prompt = self._create_supervisor_prompt(context)
            
            # Call supervisor AI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_supervisor_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent evaluation
                max_tokens=800
            )
            
            supervisor_response = response.choices[0].message.content
            
            # Parse supervisor response
            evaluation = self._parse_supervisor_response(
                supervisor_response, individual_results, consensus,
                technical_signals, news_sentiment, fundamentals
            )
            
            return evaluation
            
        except Exception as e:
            print(f"Supervisor AI error: {str(e)[:100]}")
            return self._rule_based_evaluation(
                individual_results, consensus, technical_signals, news_sentiment, fundamentals
            )
    
    def _prepare_supervisor_context(
        self, ticker: str, individual_results: Dict, consensus: Dict,
        technical_signals: Dict, news_sentiment: Dict, fundamentals: Dict, stock_info: Dict
    ) -> str:
        """Prepare comprehensive context for supervisor"""
        
        # Individual AI results summary
        ai_summary = []
        for model_name, result in individual_results.items():
            if result:
                ai_summary.append(
                    f"{model_name}: {result.get('recommendation', 'N/A')} "
                    f"({result.get('confidence', 0)}%) - {result.get('reasoning', 'N/A')[:100]}"
                )
        
        # Technical signals summary
        rsi = technical_signals.get('rsi', 50)
        trend = technical_signals.get('direction', 'unknown')
        trend_strength = technical_signals.get('strength', 0)
        
        # News sentiment summary
        news_count = news_sentiment.get('news_count', 0)
        sentiment_label = news_sentiment.get('sentiment_label', 'neutral')
        impact_score = news_sentiment.get('impact_score', 0)
        
        # Fundamentals summary
        pe_ratio = fundamentals.get('pe_ratio', None)
        revenue_growth = fundamentals.get('revenue_growth_yoy', None)
        earnings_growth = fundamentals.get('earnings_growth_yoy', None)
        debt_to_equity = fundamentals.get('debt_to_equity', None)
        
        # Format fundamentals for display
        pe_str = f"{pe_ratio:.1f}" if pe_ratio else 'N/A'
        rev_growth_str = f"{revenue_growth:.1f}%" if revenue_growth is not None else 'N/A'
        earn_growth_str = f"{earnings_growth:.1f}%" if earnings_growth is not None else 'N/A'
        de_str = f"{debt_to_equity:.1f}%" if debt_to_equity is not None else 'N/A'
        
        context = f"""
STOCK: {ticker} ({stock_info.get('name', ticker)})
SECTOR: {stock_info.get('sector', 'Unknown')}

=== INDIVIDUAL AI PREDICTIONS ===
{chr(10).join(ai_summary) if ai_summary else 'No AI predictions available'}

=== CONSENSUS ===
Recommendation: {consensus.get('recommendation', 'N/A')}
Confidence: {consensus.get('confidence', 0)}%
Agreement: {consensus.get('agreement_percentage', 0)}%

=== TECHNICAL ANALYSIS ===
RSI: {rsi:.1f}
Trend: {trend} (Strength: {trend_strength}/100)
Price Change (5d): {technical_signals.get('price_change_5d', 0):.2f}%

=== NEWS & SENTIMENT ===
News Count (7d): {news_count}
Sentiment: {sentiment_label} (Impact: {impact_score:.1f}/100)
Positive News: {news_sentiment.get('positive_count', 0)}
Negative News: {news_sentiment.get('negative_count', 0)}

=== FUNDAMENTALS ===
P/E Ratio: {pe_str}
Revenue Growth (YoY): {rev_growth_str}
Earnings Growth (YoY): {earn_growth_str}
Debt-to-Equity: {de_str}
"""
        return context
    
    def _get_supervisor_system_prompt(self) -> str:
        """Get system prompt for supervisor AI"""
        return """You are an expert AI supervisor that evaluates other AI predictions for stock trading.

Your role:
1. Review all AI predictions and identify contradictions
2. Evaluate if predictions align with technical, fundamental, and news data
3. Flag high-risk scenarios or overconfidence
4. Provide calibrated confidence scores
5. Suggest when to WAIT vs. ACT

Focus on:
- Agreement/disagreement between AI models
- Alignment with technical indicators
- News sentiment impact
- Fundamental health
- Risk assessment
- Confidence calibration"""
    
    def _create_supervisor_prompt(self, context: str) -> str:
        """Create prompt for supervisor"""
        return f"""As an AI supervisor, evaluate these stock predictions:

{context}

Analyze:
1. Are the AI predictions consistent? Any contradictions?
2. Do predictions align with technical indicators?
3. Does news sentiment support or contradict the recommendation?
4. Are fundamentals healthy?
5. What are the risks?
6. Is the confidence level appropriate?

Provide your evaluation in this format:
CONTRADICTIONS: [List any contradictions between AIs or with data]
RISK_FLAGS: [List any red flags or concerns]
ALIGNMENT: [How well do predictions align with data?]
CONFIDENCE_CALIBRATION: [Should confidence be adjusted? Why?]
FINAL_RECOMMENDATION: [BUY/SELL/WAIT]
SUPERVISOR_CONFIDENCE: [0-100]
REASONING: [Detailed reasoning]"""
    
    def _parse_supervisor_response(
        self, response: str, individual_results: Dict, consensus: Dict,
        technical_signals: Dict, news_sentiment: Dict, fundamentals: Dict
    ) -> Dict:
        """Parse supervisor AI response"""
        evaluation = {
            'contradictions': [],
            'risk_flags': [],
            'alignment_score': 50,
            'confidence_adjustment': 0,
            'supervisor_recommendation': consensus.get('recommendation', 'WAIT'),
            'supervisor_confidence': consensus.get('confidence', 50),
            'supervisor_reasoning': response,
            'final_recommendation': consensus.get('recommendation', 'WAIT'),
            'final_confidence': consensus.get('confidence', 50)
        }
        
        # Extract contradictions
        if "CONTRADICTIONS:" in response:
            lines = response.split("CONTRADICTIONS:")[1].split("\n")
            for line in lines[:5]:  # First 5 lines
                line = line.strip()
                if line and not line.startswith(("RISK_FLAGS", "ALIGNMENT", "CONFIDENCE")):
                    evaluation['contradictions'].append(line)
        
        # Extract risk flags
        if "RISK_FLAGS:" in response:
            lines = response.split("RISK_FLAGS:")[1].split("\n")
            for line in lines[:5]:
                line = line.strip()
                if line and not line.startswith(("ALIGNMENT", "CONFIDENCE", "FINAL")):
                    evaluation['risk_flags'].append(line)
        
        # Extract final recommendation
        if "FINAL_RECOMMENDATION:" in response:
            rec_line = [l for l in response.split('\n') if 'FINAL_RECOMMENDATION:' in l]
            if rec_line:
                rec_text = rec_line[0].upper()
                if 'BUY' in rec_text:
                    evaluation['supervisor_recommendation'] = 'BUY'
                elif 'SELL' in rec_text:
                    evaluation['supervisor_recommendation'] = 'SELL'
                else:
                    evaluation['supervisor_recommendation'] = 'WAIT'
        
        # Extract supervisor confidence
        if "SUPERVISOR_CONFIDENCE:" in response:
            conf_line = [l for l in response.split('\n') if 'SUPERVISOR_CONFIDENCE:' in l]
            if conf_line:
                try:
                    conf = int(''.join(filter(str.isdigit, conf_line[0])))
                    evaluation['supervisor_confidence'] = min(100, max(0, conf))
                except:
                    pass
        
        # Calculate final recommendation (supervisor can override consensus)
        if evaluation['risk_flags']:
            # If risk flags, reduce confidence
            evaluation['final_confidence'] = max(0, evaluation['supervisor_confidence'] - 10)
            if len(evaluation['risk_flags']) >= 3:
                evaluation['final_recommendation'] = 'WAIT'
        else:
            evaluation['final_recommendation'] = evaluation['supervisor_recommendation']
            evaluation['final_confidence'] = evaluation['supervisor_confidence']
        
        return evaluation
    
    def _rule_based_evaluation(
        self, individual_results: Dict, consensus: Dict,
        technical_signals: Dict, news_sentiment: Dict, fundamentals: Dict
    ) -> Dict:
        """Rule-based fallback when AI supervisor unavailable"""
        contradictions = []
        risk_flags = []
        
        # Check for contradictions in AI predictions
        recommendations = [r.get('recommendation') for r in individual_results.values() if r]
        if len(set(recommendations)) > 2:  # More than 2 different recommendations
            contradictions.append("High disagreement between AI models")
        
        # Check news sentiment vs. consensus
        sentiment_label = news_sentiment.get('sentiment_label', 'neutral')
        consensus_rec = consensus.get('recommendation', 'WAIT')
        if consensus_rec == 'BUY' and sentiment_label == 'negative':
            risk_flags.append("Negative news sentiment contradicts BUY recommendation")
        elif consensus_rec == 'SELL' and sentiment_label == 'positive':
            risk_flags.append("Positive news sentiment contradicts SELL recommendation")
        
        # Check fundamentals
        debt_to_equity = fundamentals.get('debt_to_equity', None)
        if debt_to_equity and debt_to_equity > 100:
            risk_flags.append("High debt-to-equity ratio (>100%)")
        
        earnings_growth = fundamentals.get('earnings_growth_yoy', None)
        if earnings_growth and earnings_growth < -20:
            risk_flags.append("Significant earnings decline (>20%)")
        
        # Calculate alignment score
        alignment_score = 50
        if not contradictions:
            alignment_score += 20
        if not risk_flags:
            alignment_score += 20
        if sentiment_label == 'positive' and consensus_rec == 'BUY':
            alignment_score += 10
        
        # Final recommendation
        final_rec = consensus.get('recommendation', 'WAIT')
        final_conf = consensus.get('confidence', 50)
        
        if risk_flags:
            final_conf = max(0, final_conf - 15)
            if len(risk_flags) >= 2:
                final_rec = 'WAIT'
        
        return {
            'contradictions': contradictions,
            'risk_flags': risk_flags,
            'alignment_score': alignment_score,
            'confidence_adjustment': 0,
            'supervisor_recommendation': final_rec,
            'supervisor_confidence': final_conf,
            'supervisor_reasoning': f"Rule-based evaluation: {len(contradictions)} contradictions, {len(risk_flags)} risk flags",
            'final_recommendation': final_rec,
            'final_confidence': final_conf
        }

