# AI Analysis Documentation

## 📁 Files

- **`core/analysis/ai_analyzer.py`** - AI-powered stock analysis using OpenAI (single model)
- **`multi_ai_scanner.py`** - ⭐ **Multi-AI ensemble scanner** (OpenAI + Hugging Face consensus)

## 🎯 Purpose

The AI analyzer uses OpenAI GPT models to provide context-aware stock analysis and buy/sell recommendations. It combines technical indicators, fundamentals, and market context to generate intelligent trading recommendations.

**For higher accuracy**, use the **Multi-AI Scanner** (`multi_ai_scanner.py`) which combines multiple AI models (OpenAI + Hugging Face) for consensus-based recommendations. See [Multi-AI Scanner section](#with-multi-ai-scanner) below.

## 🚀 Quick Start

### Basic Usage

```python
from core.analysis import StockAIAnalyzer
from core.data import fetch_stock_data, get_stock_info
from core.analysis import calculate_all_indicators, get_current_signals

# Initialize AI analyzer
analyzer = StockAIAnalyzer(api_key="your-api-key")  # Or set OPENAI_API_KEY env var

# Fetch data
df = fetch_stock_data("AAPL", period="3mo")
df_indicators = calculate_all_indicators(df)
signals = get_current_signals(df_indicators)
stock_info = get_stock_info("AAPL")

# Analyze with AI
analysis = analyzer.analyze_stock(
    ticker="AAPL",
    stock_info=stock_info,
    technical_signals=signals,
    price_data_summary={}
)

# Access results
print(f"Recommendation: {analysis['recommendation']}")
print(f"Confidence: {analysis['confidence']}%")
print(f"Reasoning: {analysis['reasoning']}")
```

### Using Environment Variable

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

Then use without passing API key:

```python
analyzer = StockAIAnalyzer()  # Uses OPENAI_API_KEY from environment
```

## 🔧 Configuration

### Model Selection

```python
# Default: gpt-4o-mini (fast, cost-effective)
analyzer = StockAIAnalyzer(model="gpt-4o-mini")

# More powerful: gpt-4 (slower, more expensive)
analyzer = StockAIAnalyzer(model="gpt-4")

# Faster: gpt-3.5-turbo (less capable)
analyzer = StockAIAnalyzer(model="gpt-3.5-turbo")
```

**Model Comparison**:
- **gpt-4o-mini**: Recommended - Good balance of speed, cost, and quality
- **gpt-4**: Best quality but slower and more expensive
- **gpt-3.5-turbo**: Fastest and cheapest but less capable

## 📊 Analysis Output

The `analyze_stock()` method returns a dictionary with:

```python
{
    'recommendation': 'BUY',           # 'BUY', 'SELL', or 'WAIT'
    'confidence': 85,                  # 0-100 (higher = more confident)
    'reasoning': 'Strong upward trend...',  # AI-generated explanation
    'upside_potential': 'High',        # 'Low', 'Medium', or 'High'
    'risk_level': 'Low',              # 'Low', 'Medium', or 'High'
    'ai_analysis': 'Full AI response...',  # Complete AI response
    'technical_score': 82.5           # Technical analysis score (0-100)
}
```

### Recommendation Values

- **BUY**: Strong buy opportunity with high confidence
- **SELL**: Sell signal (overbought or bearish conditions)
- **WAIT**: Uncertain conditions, wait for better setup

### Confidence Levels

- **80-100**: Very high confidence
- **60-79**: High confidence
- **40-59**: Medium confidence
- **0-39**: Low confidence

### Upside Potential

- **High**: Significant upside potential (>10% expected)
- **Medium**: Moderate upside (5-10% expected)
- **Low**: Limited upside (<5% expected)

### Risk Levels

- **Low**: Low risk trade (strong trend, good setup)
- **Medium**: Moderate risk (mixed signals)
- **High**: High risk (weak setup, uncertain conditions)

## 🔍 How It Works

### 1. Context Preparation

The AI analyzer prepares comprehensive context including:

- **Stock Information**: Ticker, name, sector, industry
- **Current Price**: Latest price and price changes (1d, 5d, 20d)
- **Technical Indicators**: RSI, trend, EMA, MACD, Bollinger Bands
- **Market Data**: 52-week highs/lows, P/E ratio, market cap
- **Demand/Supply Zones**: Key support and resistance levels

### 2. AI Prompt

The system sends a structured prompt to OpenAI:

```
Analyze this stock for a SCALPING/SHORT-TERM trading opportunity:

STOCK: AAPL (Apple Inc.)
SECTOR: Technology
CURRENT PRICE: $180.50
PRICE CHANGE (1d): 1.2%
PRICE CHANGE (5d): 3.5%
TECHNICAL INDICATORS:
- RSI: 45.3 (oversold)
- Trend: up (Strength: 75.5/100)
- EMA Cross: bullish
- MACD Signal: bullish
...

Based on the technical indicators and current market conditions, 
should I BUY this stock RIGHT NOW for short-term gains?

Provide your analysis in this format:
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
REASONING: [Brief explanation]
UPSIDE_POTENTIAL: [Low/Medium/High]
RISK_LEVEL: [Low/Medium/High]
```

### 3. Response Parsing

The system parses the AI response to extract:
- Recommendation (BUY/SELL/WAIT)
- Confidence score (0-100)
- Reasoning (explanation)
- Upside potential (Low/Medium/High)
- Risk level (Low/Medium/High)

### 4. Fallback Analysis

If AI is unavailable, the system falls back to rule-based technical analysis:

```python
# Fallback uses technical score
technical_score = calculate_technical_score(signals)

if technical_score >= 70 and trend == 'up':
    recommendation = 'BUY'
elif technical_score < 40:
    recommendation = 'WAIT'
else:
    recommendation = 'CONSIDER BUY'
```

## 📈 Technical Score Calculation

The fallback analysis calculates a technical score (0-100) based on:

- **RSI Contribution** (0-15 points)
  - RSI 30-70: +10 points
  - RSI < 30 (oversold): +15 points
  
- **Trend Contribution** (0-20 points)
  - Uptrend: +min(20, strength/5) points
  
- **Price Change Contribution** (0-15 points)
  - Positive 5d change: +min(15, change_5d) points
  
- **EMA Cross Contribution** (0-10 points)
  - Bullish EMA cross: +10 points
  
- **MACD Contribution** (0-10 points)
  - Bullish MACD: +10 points

**Base Score**: 50 points

## 🔄 Batch Analysis

### Analyze Multiple Stocks

```python
from core.analysis import StockAIAnalyzer

analyzer = StockAIAnalyzer()

stocks_data = [
    {
        'ticker': 'AAPL',
        'stock_info': {...},
        'technical_signals': {...},
        'price_data_summary': {...}
    },
    {
        'ticker': 'MSFT',
        'stock_info': {...},
        'technical_signals': {...},
        'price_data_summary': {...}
    }
]

# Batch analyze
results = analyzer.batch_analyze(stocks_data)

for result in results:
    print(f"{result['ticker']}: {result['recommendation']} ({result['confidence']}%)")
```

## ⚙️ Advanced Usage

### Custom System Prompt

The system uses a default system prompt, but you can modify it by editing `_get_system_prompt()`:

```python
def _get_system_prompt(self) -> str:
    return """You are an expert stock analyst specializing in identifying 
    high-potential tech stocks and rising stocks with upside potential. 
    Analyze the provided stock data and technical indicators to determine 
    if this is a good buy opportunity RIGHT NOW.
    
    Focus on:
    1. Technical momentum and trend strength
    2. Price action and recent performance
    3. Relative strength compared to market
    4. Risk/reward ratio
    5. Entry timing
    
    Provide a clear BUY, SELL, or WAIT recommendation with confidence level 
    (0-100) and brief reasoning."""
```

### Custom Analysis Prompt

Modify `_create_analysis_prompt()` to change how the AI analyzes stocks:

```python
def _create_analysis_prompt(self, context: str) -> str:
    return f"""Analyze this stock for a SCALPING/SHORT-TERM trading opportunity:

{context}

Based on the technical indicators and current market conditions, 
should I BUY this stock RIGHT NOW for short-term gains?

Provide your analysis in this format:
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
REASONING: [Brief explanation]
UPSIDE_POTENTIAL: [Low/Medium/High]
RISK_LEVEL: [Low/Medium/High]"""
```

## 🛡️ Error Handling

The analyzer includes robust error handling:

1. **API Errors**: Falls back to technical analysis
2. **Missing API Key**: Uses fallback analysis
3. **Network Issues**: Returns fallback analysis
4. **Invalid Response**: Parses what it can, uses defaults

```python
try:
    analysis = analyzer.analyze_stock(...)
except Exception as e:
    print(f"Error: {e}")
    # System automatically falls back to technical analysis
```

## 💰 Cost Considerations

### API Costs (OpenAI)

- **gpt-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **gpt-4**: ~$30 per 1M input tokens, ~$60 per 1M output tokens
- **gpt-3.5-turbo**: ~$0.50 per 1M input tokens, ~$1.50 per 1M output tokens

**Typical Cost per Analysis**:
- **gpt-4o-mini**: ~$0.001-0.002 per stock
- **gpt-4**: ~$0.01-0.02 per stock
- **gpt-3.5-turbo**: ~$0.0005-0.001 per stock

### Cost Optimization

1. **Use gpt-4o-mini**: Best balance of cost and quality
2. **Use Hugging Face Models**: Free (no API costs) - use `--preset open-hf` or `--models hf:finance-chat`
3. **Mix Models**: Combine free HF models with OpenAI for cost savings
4. **Batch Analysis**: Analyze multiple stocks in one API call
5. **Cache Results**: Store results to avoid re-analyzing
6. **Rate Limiting**: Built-in delays prevent excessive API calls

### Multi-AI Scanner Cost Comparison

- **OpenAI-only**: ~$0.003-0.006 per stock (3 models)
- **Hugging Face-only**: $0 per stock (free, but requires GPU/CPU resources)
- **Mixed (OpenAI + HF)**: ~$0.001-0.002 per stock (combines free HF with paid OpenAI)
- **Finance-wide preset**: ~$0.002-0.004 per stock (3 OpenAI + 7 HF models)

## 🔗 Integration Examples

### With Stock Scanner

```python
from scanners import AgenticStockScanner

scanner = AgenticStockScanner(openai_api_key="your-key")
results = scanner.scan_stocks(focus="tech", period="3mo")

# Results include AI analysis for each stock
for stock in results['recommendations']:
    print(f"{stock['ticker']}: {stock['recommendation']} ({stock['confidence']}%)")
```

### With Multi-AI Scanner

The **Multi-AI Scanner** (`multi_ai_scanner.py`) combines multiple AI models (OpenAI + Hugging Face) for consensus-based recommendations with higher accuracy.

#### Use ALL Models (OpenAI + Hugging Face)

**Using Presets (Recommended):**

```bash
# Diversified preset (3 OpenAI + 3 Hugging Face models)
python multi_ai_scanner.py --preset diversified-6

# Finance-wide preset (3 OpenAI + 7 Hugging Face finance models)
python multi_ai_scanner.py --preset finance-wide

# With GPU (faster for Hugging Face models)
python multi_ai_scanner.py --preset finance-wide --use-gpu
```

**Custom Model List:**

```bash
# Mix OpenAI and Hugging Face models manually
python multi_ai_scanner.py --models gpt-4o-mini,gpt-4,gpt-3.5-turbo,hf:mistral-7b,hf:llama-2-7b,hf:zephyr-7b

# Finance-focused mix
python multi_ai_scanner.py --models gpt-4o-mini,gpt-4,hf:finance-chat,hf:llama-open-finance-8b,hf:qwen-open-finance-r-8b

# With GPU
python multi_ai_scanner.py --models gpt-4o-mini,gpt-4,hf:mistral-7b,hf:llama-2-7b --use-gpu
```

#### Use ONLY Hugging Face Models

**Using Preset:**

```bash
# Open HF preset (single finance model, no auth needed)
python multi_ai_scanner.py --preset open-hf

# With GPU
python multi_ai_scanner.py --preset open-hf --use-gpu
```

**Custom Hugging Face Models:**

```bash
# Single Hugging Face model
python multi_ai_scanner.py --models hf:finance-chat

# Multiple Hugging Face models
python multi_ai_scanner.py --models hf:mistral-7b,hf:llama-2-7b,hf:zephyr-7b

# Finance-focused Hugging Face models
python multi_ai_scanner.py --models hf:finance-chat,hf:llama-open-finance-8b,hf:qwen-open-finance-r-8b,hf:fin-o1-14b

# With GPU (recommended for faster inference)
python multi_ai_scanner.py --models hf:mistral-7b,hf:llama-2-7b,hf:zephyr-7b --use-gpu

# Using full model names
python multi_ai_scanner.py --models hf:mistralai/Mistral-7B-Instruct-v0.2
python multi_ai_scanner.py --models hf:meta-llama/Llama-2-7b-chat-hf
```

#### Use ONLY OpenAI Models

**Using Preset (Fastest):**

```bash
# OpenAI-only preset (3 models: gpt-4o-mini, gpt-4, gpt-3.5-turbo)
python multi_ai_scanner.py --preset openai-only
```

**Custom OpenAI Models:**

```bash
# Single OpenAI model
python multi_ai_scanner.py --models gpt-4o-mini

# Multiple OpenAI models
python multi_ai_scanner.py --models gpt-4o-mini,gpt-4,gpt-3.5-turbo

# Just GPT-4 models
python multi_ai_scanner.py --models gpt-4o-mini,gpt-4
```

#### Available Hugging Face Model Shortcuts

When using `hf:` prefix, you can use these shortcuts:

```bash
# General models
hf:mistral-7b          # mistralai/Mistral-7B-Instruct-v0.2
hf:llama-2-7b         # meta-llama/Llama-2-7b-chat-hf
hf:zephyr-7b          # HuggingFaceH4/zephyr-7b-beta
hf:gemma-7b           # google/gemma-7b-it
hf:phi-2              # microsoft/phi-2
hf:tinyllama          # TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Finance-specialized models
hf:finance-chat       # AdaptLLM/finance-chat
hf:llama-open-finance-8b    # DragonLLM/Llama-Open-Finance-8B
hf:qwen-open-finance-r-8b   # DragonLLM/Qwen-Open-Finance-R-8B
hf:fin-o1-14b         # TheFinAI/Fin-o1-14B
```

#### Additional Multi-AI Scanner Options

```bash
# Serial model execution (less CPU/RAM load)
python multi_ai_scanner.py --preset diversified-6 --serial-models

# Higher confidence thresholds
python multi_ai_scanner.py --preset finance-wide --min-consensus 75 --min-score 70

# Focus on specific stocks
python multi_ai_scanner.py --preset diversified-6 --tickers AAPL,MSFT,NVDA

# Focus on tech stocks
python multi_ai_scanner.py --preset finance-wide --focus tech
```

#### Quick Reference Table

| Command | Models Used | Best For |
|---------|-------------|----------|
| `--preset openai-only` | 3 OpenAI models | Fastest, no downloads |
| `--preset open-hf` | 1 HF model | No API costs, single model |
| `--preset diversified-6` | 3 OpenAI + 3 HF | Balanced consensus |
| `--preset finance-wide` | 3 OpenAI + 7 HF finance | Maximum accuracy, finance-focused |
| `--models hf:mistral-7b,hf:llama-2-7b` | 2 HF models | HF-only, custom selection |
| `--models gpt-4o-mini,gpt-4` | 2 OpenAI models | OpenAI-only, custom selection |

#### Programmatic Usage

```python
from multi_ai_scanner import MultiAIScanner

# All models (OpenAI + Hugging Face)
scanner = MultiAIScanner(preset="diversified-6")
results = scanner.scan_stocks(tickers=["AAPL", "MSFT"])

# Only Hugging Face
scanner = MultiAIScanner(models=["hf:finance-chat", "hf:mistral-7b"], use_gpu=True)
results = scanner.scan_stocks(tickers=["AAPL"])

# Only OpenAI
scanner = MultiAIScanner(preset="openai-only")
results = scanner.scan_stocks(tickers=["AAPL", "MSFT"])
```

**Note**: 
- First-time Hugging Face downloads can take several minutes (cached after first use)
- Some HF models require authentication: run `huggingface-cli login`
- GPU usage (`--use-gpu`) requires CUDA for faster inference
- Serial execution (`--serial-models`) reduces CPU/RAM usage

### Standalone Usage

```python
from core.analysis import StockAIAnalyzer
from core.data import fetch_stock_data, get_stock_info
from core.analysis import calculate_all_indicators, get_current_signals

# Setup
analyzer = StockAIAnalyzer()
df = fetch_stock_data("AAPL", period="3mo")
df_indicators = calculate_all_indicators(df)
signals = get_current_signals(df_indicators)
info = get_stock_info("AAPL")

# Analyze
analysis = analyzer.analyze_stock("AAPL", info, signals, {})

# Use results
if analysis['recommendation'] == 'BUY' and analysis['confidence'] >= 70:
    print(f"✅ Strong buy signal: {analysis['reasoning']}")
```

## 📚 Related Files

- **`core/analysis/technical_analyzer.py`** - Technical indicators (input for AI)
- **`core/data/stock_fetcher.py`** - Stock data fetching
- **`core/analysis/huggingface_analyzer.py`** - Hugging Face model integration
- **`scanners/agentic_scanner.py`** - Uses AI analyzer (single OpenAI model)
- **`multi_ai_scanner.py`** - ⭐ **Multi-AI ensemble** (OpenAI + Hugging Face consensus)

## 💡 Best Practices

1. **Set API Key**: Use environment variable for security
2. **Use gpt-4o-mini**: Best balance for most use cases
3. **Combine with Technical Analysis**: AI + technical = better results
4. **Monitor Costs**: Track API usage to avoid surprises
5. **Handle Errors**: Always have fallback (automatic in this system)
6. **Cache Results**: Store analysis results to avoid re-analyzing

## ⚠️ Important Notes

1. **Not Financial Advice**: AI analysis is for informational purposes only
2. **API Costs**: Monitor your OpenAI usage and costs
3. **Rate Limits**: OpenAI has rate limits (system includes delays)
4. **Fallback Available**: System works without AI (uses technical analysis)
5. **Data Quality**: Better input data = better AI analysis

## 🔧 Troubleshooting

### "OpenAI library not installed"

```bash
pip install openai
```

### "Could not initialize OpenAI client"

- Check API key is correct
- Verify API key has credits
- Check internet connection

### "AI analysis will be limited"

- API key not set or invalid
- System falls back to technical analysis automatically

### High API Costs

- Use `gpt-4o-mini` instead of `gpt-4`
- Use Hugging Face models (free): `python multi_ai_scanner.py --preset open-hf`
- Mix free HF models with OpenAI: `python multi_ai_scanner.py --models gpt-4o-mini,hf:finance-chat`
- Reduce number of stocks analyzed
- Cache results to avoid re-analyzing

### Hugging Face Model Issues

**"Model requires Hugging Face authentication"**

```bash
# Login to Hugging Face
huggingface-cli login

# Then use the model
python multi_ai_scanner.py --models hf:llama-2-7b
```

**"transformers library not installed"**

```bash
pip install transformers torch
```

**"GPU not available"**

- Hugging Face models will use CPU (slower but works)
- Install CUDA toolkit for GPU support
- Or use `--serial-models` to reduce CPU load

**"Model download stuck"**

- First download can take 10-30 minutes depending on model size
- Press Ctrl+C once and rerun (will resume from cache)
- Models are cached in `~/.cache/huggingface/hub`

---

**Remember**: AI analysis enhances technical analysis but doesn't guarantee profits. Always use proper risk management and do your own research!

