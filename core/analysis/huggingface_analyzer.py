"""
Hugging Face AI Analyzer - Uses Hugging Face models for stock analysis

Supports various Hugging Face models:
- Meta-Llama models (llama-2, llama-3, etc.)
- Mistral models
- Zephyr models
- Other instruction-tuned models via Hugging Face Transformers

Usage:
    from huggingface_analyzer import HuggingFaceAnalyzer
    
    analyzer = HuggingFaceAnalyzer(model_name="microsoft/DialoGPT-medium")
    result = analyzer.analyze_stock(...)
"""

import os
from typing import Dict, Optional
import warnings

# Load environment variables from .env.local or .env for HF_TOKEN
try:
    from dotenv import load_dotenv
    # Try .env.local first, then fall back to .env
    load_dotenv('.env.local')
    load_dotenv('.env')  # Fallback
except ImportError:
    pass  # python-dotenv not installed, skip silently

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: transformers library not installed. Install with: pip install transformers torch")


class HuggingFaceAnalyzer:
    """
    AI-powered stock analyzer using Hugging Face models
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 use_gpu: bool = False, max_length: int = 512):
        """
        Initialize Hugging Face analyzer
        
        Args:
            model_name: Hugging Face model identifier
                       Examples:
                       - "mistralai/Mistral-7B-Instruct-v0.2"
                       - "meta-llama/Llama-2-7b-chat-hf"
                    #    - "HuggingFaceH4/zephyr-7b-beta" # failed
                       - "google/gemma-7b-it"
            use_gpu: Whether to use GPU if available
            max_length: Maximum token length for generation
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        
        # Check GPU availability
        if use_gpu and not torch.cuda.is_available():
            print(f"⚠️ GPU requested but CUDA not available. Using CPU instead.")
        elif use_gpu:
            print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
        
        try:
            print(f"🤗 Loading Hugging Face model: {model_name}...")
            self._load_model()
            self.is_loaded = True
        except Exception as e:
            error_msg = str(e)
            if "gated repo" in error_msg.lower() or "401" in error_msg or "restricted" in error_msg.lower():
                print(f"❌ Model requires Hugging Face authentication: {model_name}")
                print(f"   Visit https://huggingface.co/{model_name.split('/')[0]}/{model_name.split('/')[1] if '/' in model_name else ''}")
                print(f"   Then run: huggingface-cli login")
            else:
                print(f"❌ Failed to load model {model_name}: {error_msg[:200]}")
            self.model = None
            self.is_loaded = False
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer"""
        device = "cuda" if self.use_gpu else "cpu"
        
        try:
            # Try to load as a text generation pipeline (easier to use)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device_map="auto" if self.use_gpu else None,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                trust_remote_code=True
            )
            actual_device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            print(f"✅ Loaded {self.model_name} on {actual_device}")
        except Exception as e:
            error_str = str(e)
            # Don't try manual load for authentication errors
            if "gated repo" in error_str.lower() or "401" in error_str or "restricted" in error_str.lower():
                raise
            print(f"⚠️ Pipeline loading failed, trying manual load: {error_str[:150]}")
            try:
                # Fallback to manual loading
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if self.use_gpu else None,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    trust_remote_code=True
                )
                actual_device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
                print(f"✅ Loaded {self.model_name} on {actual_device}")
            except Exception as e2:
                print(f"❌ Failed to load model: {str(e2)[:200]}")
                raise
    
    def analyze_stock(self, ticker: str, stock_info: Dict, technical_signals: Dict,
                     price_data_summary: Dict) -> Dict:
        """
        Analyze stock using Hugging Face model
        
        Args:
            ticker: Stock ticker
            stock_info: Stock information
            technical_signals: Technical analysis signals
            price_data_summary: Price data summary
        
        Returns:
            Dictionary with analysis result
        """
        if not self.pipeline and not self.model:
            return self._fallback_analysis(ticker, stock_info, technical_signals, price_data_summary)
        
        try:
            # Prepare context
            context = self._prepare_context(ticker, stock_info, technical_signals, price_data_summary)
            
            # Create prompt
            prompt = self._create_analysis_prompt(context)
            
            # Generate response
            if self.pipeline:
                response = self._generate_with_pipeline(prompt)
            else:
                response = self._generate_with_model(prompt)
            
            # Parse response
            analysis = self._parse_ai_response(response, technical_signals)
            return analysis
            
        except Exception as e:
            print(f"Error in Hugging Face analysis for {ticker}: {str(e)}")
            return self._fallback_analysis(ticker, stock_info, technical_signals, price_data_summary)
    
    def _generate_with_pipeline(self, prompt: str) -> str:
        """Generate response using pipeline"""
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                return_full_text=False
            )
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            return str(result)
        except Exception as e:
            print(f"Pipeline generation error: {e}")
            return ""
    
    def _generate_with_model(self, prompt: str) -> str:
        """Generate response using model directly"""
        if not self.tokenizer or not self.model:
            return ""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            
            if self.use_gpu:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
        except Exception as e:
            print(f"Model generation error: {e}")
            return ""
    
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
    
    def _create_analysis_prompt(self, context: str) -> str:
        """Create analysis prompt for Hugging Face model"""
        # Use instruction format that works well with most instruction-tuned models
        prompt = f"""<s>[INST] You are an expert stock analyst. Analyze this stock for a SHORT-TERM trading opportunity.

{context}

Based on the technical indicators and current market conditions, should I BUY this stock RIGHT NOW for short-term gains?

Provide your analysis in this format:
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
REASONING: [Brief explanation]
UPSIDE_POTENTIAL: [Low/Medium/High]
RISK_LEVEL: [Low/Medium/High] [/INST]"""
        return prompt
    
    def _parse_ai_response(self, ai_response: str, technical_signals: Dict) -> Dict:
        """Parse AI response into structured format"""
        recommendation = "WAIT"
        confidence = 50
        reasoning = ai_response
        upside_potential = "Medium"
        risk_level = "Medium"
        
        # Extract recommendation
        if "RECOMMENDATION:" in ai_response:
            lines = ai_response.split('\n')
            for line in lines:
                if 'RECOMMENDATION:' in line:
                    if 'BUY' in line.upper():
                        recommendation = "BUY"
                    elif 'SELL' in line.upper():
                        recommendation = "SELL"
                    break
        
        # Extract confidence
        if "CONFIDENCE:" in ai_response:
            lines = ai_response.split('\n')
            for line in lines:
                if 'CONFIDENCE:' in line:
                    try:
                        confidence = int(''.join(filter(str.isdigit, line)))
                        confidence = max(0, min(100, confidence))  # Clamp 0-100
                    except:
                        pass
                    break
        
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
            lines = ai_response.split('\n')
            for line in lines:
                if 'UPSIDE_POTENTIAL:' in line:
                    if 'HIGH' in line.upper():
                        upside_potential = "High"
                    elif 'LOW' in line.upper():
                        upside_potential = "Low"
                    break
        
        if "RISK_LEVEL:" in ai_response:
            lines = ai_response.split('\n')
            for line in lines:
                if 'RISK_LEVEL:' in line:
                    if 'HIGH' in line.upper():
                        risk_level = "High"
                    elif 'LOW' in line.upper():
                        risk_level = "Low"
                    break
        
        # Calculate technical score as backup
        technical_score = self._calculate_technical_score(technical_signals)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': reasoning or f"Analysis from {self.model_name}",
            'upside_potential': upside_potential,
            'risk_level': risk_level,
            'technical_score': technical_score,
            'ai_analysis': ai_response,
            'model_name': self.model_name
        }
    
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
    
    def _fallback_analysis(self, ticker: str, stock_info: Dict,
                          technical_signals: Dict, price_data_summary: Dict) -> Dict:
        """Fallback analysis when model is not available"""
        technical_score = self._calculate_technical_score(technical_signals)
        rsi = technical_signals.get('rsi', 50)
        trend = technical_signals.get('direction', 'unknown')
        
        if technical_score >= 70:
            recommendation = "BUY"
            confidence = min(75, technical_score)
        elif technical_score <= 30:
            recommendation = "SELL"
            confidence = 60
        else:
            recommendation = "WAIT"
            confidence = 50
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': f"Technical analysis only (Hugging Face model {self.model_name} unavailable): {trend} trend, RSI {rsi:.1f}",
            'upside_potential': "High" if technical_score >= 70 else ("Medium" if technical_score >= 50 else "Low"),
            'risk_level': "Low" if technical_score >= 70 else ("Medium" if technical_score >= 50 else "High"),
            'technical_score': technical_score,
            'model_name': self.model_name
        }


# Recommended models for stock analysis (smaller, faster models)
RECOMMENDED_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "gemma-7b": "google/gemma-7b-it",
    "phi-2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Very small, fast
    # Finance-specialized LLMs
    "finance-chat": "AdaptLLM/finance-chat",
    "llama-open-finance-8b": "DragonLLM/Llama-Open-Finance-8B",
    "qwen-open-finance-r-8b": "DragonLLM/Qwen-Open-Finance-R-8B",
    "fin-o1-14b": "TheFinAI/Fin-o1-14B"
}

