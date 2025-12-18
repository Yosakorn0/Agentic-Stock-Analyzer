# Agentic AI Stock Scanner

An intelligent AI-powered system that scans tech stocks and rising stocks to identify high-potential buy opportunities RIGHT NOW.

## 🎯 Features

- **AI-Powered Analysis**: Uses OpenAI GPT models to analyze stocks with context-aware reasoning
- **Multi-AI Ensemble (OpenAI + Hugging Face)**: Optional consensus mode with a lightweight preset to minimize downloads and resource usage
- **Technical Analysis**: Comprehensive technical indicators (RSI, MACD, EMA, Bollinger Bands, etc.)
- **Smart Screening**: Filters stocks by tech sector, rising momentum, oversold conditions, and more
- **Real-Time Scanning**: Scans multiple stocks simultaneously with rate limiting
- **Buy Recommendations**: Generates actionable buy/sell/wait recommendations with confidence scores
- **Entry Price Suggestions**: Provides specific buy prices based on demand zones and technical analysis
- **Demand & Supply Zones**: Identifies key support (demand) and resistance (supply) levels
- **Strong Zone Analysis**: Calculates zone strength based on touches, volume, and bounce/rejection patterns
- **Gold Scalping Support**: Specialized scalping analysis with intraday intervals (5m, 15m, 30m) and optimal entry/exit prices
- **Risk Management**: Suggests stop-loss and take-profit levels with risk/reward ratios
- **Risk Assessment**: Evaluates upside potential and risk levels for each opportunity

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Tip: The multi-AI ensemble uses a lightweight preset (`open-hf`) with a single Hugging Face model (`hf:mistral-7b`) to minimize downloads and memory.

### 2. Set OpenAI API Key (Optional but Recommended)

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

**Note**: The system works without OpenAI API (uses technical analysis fallback), but AI analysis provides much better recommendations.

### 3. Run the Scanner

**Note**: Make sure you're in the `ai-stock-scanner` directory when running the modular version.

**You have TWO options:**

#### Option A: Single Combined File (Recommended for Easy Use)
```bash
# Scan all stocks (tech + rising)
python stock_scanner.py

# Focus on tech stocks only
python stock_scanner.py --focus tech

# Focus on rising stocks only
python stock_scanner.py --focus rising

# Show up to 20 BUY recommendations (if fewer qualify, fewer will be shown)
python stock_scanner.py --limit 20

# Save results to JSON file
python stock_scanner.py --save
```

#### Option B: Modular Files (For Custom Development)

**Standard Version (Sequential / Serial Processing):**
```bash
# Scan all stocks (tech + rising)
python -m scanners.agentic_scanner

# Focus on tech stocks only
python -m scanners.agentic_scanner --focus tech

# Focus on rising stocks only
python -m scanners.agentic_scanner --focus rising

# Show up to 20 BUY recommendations (if fewer qualify, fewer will be shown)
python -m scanners.agentic_scanner --limit 20

# Save results to JSON file
python -m scanners.agentic_scanner --save
```

**With Parallel Processing (Faster for Multiple Stocks):**
```python
from scanners import AgenticStockScanner

# Initialize with parallel processing
scanner = AgenticStockScanner(parallel=True, max_workers=5)

# Scan all stocks in parallel
results = scanner.scan_stocks(focus="all", period="3mo", parallel=True)

# Print recommendations
scanner.print_recommendations(limit=10)
```

**Or use CLI with parallel flag (parallel instead of serial):**
```bash
python -m scanners.agentic_scanner --parallel --workers 5
```

**Note:** Parallel processing significantly speeds up analysis when scanning many stocks, but uses more API rate limit quota. Use `max_workers` to control concurrency.

**Note**: `stock_scanner.py` is a single combined file with all functionality. Use it if you want everything in one place. The separate files (`agentic_scanner.py`, `ai_analyzer.py`, etc.) are for modular use. By default these run **serially**; enable parallel mode only if you’re comfortable with higher CPU/RAM usage.

### 4. Multi-AI Ensemble Scanner (Consensus with lightweight HF preset)

Use `multi_ai_scanner.py` to combine multiple AI models. The default `open-hf` preset now uses a single finance-tuned Hugging Face model (`hf:finance-chat`) to reduce download size and RAM/GPU usage. Add `--serial-models` to run models in **series** (less CPU/RAM load) instead of in parallel.

**Low-resource, cached HF model, serial execution (recommended on low RAM):**
```bash
python multi_ai_scanner.py --preset open-hf --serial-models
```

**OpenAI-only (no HF downloads, serial):**
```bash
python multi_ai_scanner.py --preset openai-only --serial-models
```

**Explicit single HF model (serial):**
```bash
python multi_ai_scanner.py --models hf:finance-chat --serial-models
```

**Enable GPU if available (still serial models):**
```bash
python multi_ai_scanner.py --preset open-hf --use-gpu --serial-models
```

**Parallel models (faster but heavier on RAM/CPU):**
```bash
python multi_ai_scanner.py --preset open-hf            # parallel by default
python multi_ai_scanner.py --preset open-hf --parallel-models   # explicit
```

**Optional: suppress HF progress bars/noise (per session):**
```bash
# PowerShell
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:TQDM_DISABLE = "1"   # optional
```

Notes:
- First HF download can take several minutes; subsequent runs are fast from cache (`~/.cache/huggingface/hub`).
- If a download appears stuck, press Ctrl+C once and rerun; it will resume from cache.
- Use `--serial-models` to avoid parallel model loading and lower CPU/RAM/GPU usage; use parallel mode only if you have enough resources.

## 🎯 Which Scanner Should I Use?

The project includes **four different scanner files**, each designed for different use cases. Here's a comparison to help you choose:

### Scanner Comparison Table

| Feature | `stock_scanner.py` | `my_analysis.py` | `multi_ai_scanner.py` | `high_confidence_scanner.py` |
|---------|-------------------|------------------|----------------------|------------------------------|
| **AI Models** | 1 (OpenAI) | None | Multiple (OpenAI + HF) | 1 (OpenAI) |
| **Stocks Analyzed** | Multiple | 1 | Multiple | Multiple |
| **Consensus Analysis** | ❌ No | ❌ No | ✅ Yes | ❌ No |
| **Entry Price Recommendations** | ❌ No | ❌ No | ✅ Yes | ❌ No |
| **Stop-Loss/Take-Profit** | ❌ No | ❌ No | ✅ Yes | ❌ No |
| **Agreement Percentage** | ❌ No | ❌ No | ✅ Yes | ❌ No |
| **Complexity** | Medium | Low | High | Low |
| **Best For** | Base library | Quick check | Best accuracy | Simple filtering |

### Detailed Scanner Descriptions

#### 1. `stock_scanner.py` - Core Library (All-in-One)
**Purpose:** Complete standalone scanner with all functionality in one file.

**Features:**
- ✅ Technical analysis (RSI, MACD, EMA, Bollinger Bands, etc.)
- ✅ Stock data fetching
- ✅ Single AI analyzer (OpenAI only)
- ✅ Stock screening and filtering
- ✅ Can scan multiple stocks
- ✅ Can be run standalone or imported

**Use When:**
- You want a complete solution in one file
- You need the base functionality for custom scripts
- You want to build your own tools on top of it

**Usage:**
```bash
# Scan all stocks
python stock_scanner.py

# Focus on tech stocks
python stock_scanner.py --focus tech

# Scan specific tickers
python stock_scanner.py --tickers AAPL,MSFT,NVDA
```

---

#### 2. `my_analysis.py` - Simple Single-Stock Analyzer
**Purpose:** Quick analysis for one stock at a time.

**Features:**
- ✅ Prompts for single ticker input
- ✅ Fetches data and calculates indicators
- ✅ Prints formatted signals
- ❌ No AI analysis
- ❌ Minimal output

**Use When:**
- You want a quick check on one stock
- You don't need AI recommendations
- You just want technical indicators

**Usage:**
```bash
python my_analysis.py
# Then enter: AAPL
```

---

#### 3. `multi_ai_scanner.py` - Advanced Multi-AI Ensemble Scanner ⭐ **RECOMMENDED**
**Purpose:** Uses multiple AI models for consensus-based recommendations.

**Features:**
- ✅ Multiple AI models (OpenAI + Hugging Face)
- ✅ Consensus from multiple models
- ✅ Agreement percentages and model breakdowns
- ✅ Entry price, stop-loss, take-profit recommendations
- ✅ Risk/reward ratio calculations
- ✅ Preset model bundles (`openai-only`, `open-hf`, `diversified-6`, etc.)
- ✅ Parallel or serial execution (`--serial-models`)
- ✅ Timeout controls for model loading and analysis
- ✅ Most detailed and accurate output

**Use When:**
- You want the highest accuracy with multiple AI opinions
- You need entry price recommendations
- You want consensus-based decisions
- You're making actual trading decisions

**Usage:**
```bash
# Use lightweight preset with serial execution (recommended)
python multi_ai_scanner.py --preset open-hf --serial-models

# Scan specific tickers
python multi_ai_scanner.py --preset open-hf --serial-models --tickers AAPL,MSFT,AVGO

# OpenAI-only (fast, no downloads)
python multi_ai_scanner.py --preset openai-only

# With GPU (if available)
python multi_ai_scanner.py --preset open-hf --use-gpu --serial-models
```

---

#### 4. `high_confidence_scanner.py` - High-Confidence Filter Wrapper
**Purpose:** Simple wrapper that filters for high-confidence buy opportunities.

**Features:**
- ✅ Wraps `AgenticStockScanner` from `stock_scanner.py`
- ✅ Filters for scores ≥ 60 (configurable)
- ✅ Simpler output format
- ✅ Single AI model (OpenAI)
- ✅ Focuses on BUY/CONSIDER BUY recommendations

**Use When:**
- You want a simple list of high-confidence buys
- You don't need entry prices or consensus
- You prefer a cleaner, simpler output

**Usage:**
```bash
# Scan with default 60+ score threshold
python high_confidence_scanner.py

# Higher threshold (70+)
python high_confidence_scanner.py --min-score 70

# Focus on tech stocks
python high_confidence_scanner.py --focus tech

# Scan specific tickers
python high_confidence_scanner.py --tickers AAPL,MSFT,NVDA
```

---

### Quick Decision Guide

**Choose `multi_ai_scanner.py` if:**
- ✅ You want the most accurate recommendations
- ✅ You need entry prices and stop-losses
- ✅ You want consensus from multiple AIs
- ✅ You're making trading decisions

**Choose `stock_scanner.py` if:**
- ✅ You want a complete solution in one file
- ✅ You're building custom tools
- ✅ You need the base library functionality

**Choose `high_confidence_scanner.py` if:**
- ✅ You want a simple high-confidence buy list
- ✅ You don't need entry prices
- ✅ You prefer simpler output

**Choose `my_analysis.py` if:**
- ✅ You just want to check one stock quickly
- ✅ You don't need AI recommendations
- ✅ You only want technical indicators

---

## 📊 Usage Examples

### Basic Usage

```python
from scanners import AgenticStockScanner

# Initialize scanner
scanner = AgenticStockScanner(openai_api_key="your-key")  # Optional

# Scan all stocks
results = scanner.scan_stocks(focus="all", period="3mo")

# Print top recommendations
scanner.print_recommendations(limit=10)

# Save results
scanner.save_results("my_scan_results.json")
```

### Scan Specific Stocks

```python
from scanners import AgenticStockScanner

scanner = AgenticStockScanner()

# Scan specific tickers
results = scanner.scan_stocks(
    tickers=['AAPL', 'MSFT', 'NVDA', 'TSLA'],
    period="1mo"
)

scanner.print_recommendations()
```

### Use Individual Components

#### Getting Trading Signals

There are **three ways** to use `get_current_signals()`:

**Option 1: Run the technical analyzer directly**
```bash
python core/analysis/technical_analyzer.py
```
This runs a built-in example analyzing AAPL stock.

**Option 2: Import and use in your own script** (Recommended for customization)
```python
from core.analysis import get_current_signals, calculate_all_indicators
from core.data import fetch_stock_data
from utils.format_signals import print_signals  # Optional: for formatted output

# Fetch stock data
df = fetch_stock_data("AAPL", period="3mo")

# Calculate all indicators first
df_with_indicators = calculate_all_indicators(df)

# Get current signals
signals = get_current_signals(df_with_indicators)

# Option A: Print formatted (recommended)
print_signals(signals, "AAPL")

# Option B: Access raw dictionary
print(signals)  # Returns: RSI, trend, price changes, MACD signal, etc.
```

Or use the comprehensive examples file:
```bash
# Get trading signals for a single stock
python examples/usage_examples.py signals

# Run full scanner (sequential)
python examples/usage_examples.py scan

# Run full scanner (parallel - faster)
python examples/usage_examples.py parallel

# Custom analysis template
python examples/usage_examples.py custom

# Run all examples
python examples/usage_examples.py all
```

### Gold Scalping Example

For gold scalping with intraday data and strong zone identification:
```bash
python gold_scalping_example.py
```

**Features:**
- **Intraday Intervals**: Supports 5m, 15m, 30m intervals for scalping
- **Strong Zone Identification**: Analyzes demand/supply zones with strength scoring based on:
  - Number of touches
  - Recent activity
  - Volume at zone levels
  - Bounce/rejection strength
- **Best Buy/Sell Prices**: Automatically identifies optimal entry/exit points
- **Risk/Reward Analysis**: Calculates R:R ratios for scalping trades
- **Quick Scalping Signals**: Provides immediate action recommendations

**Usage:**
```bash
# Interactive mode (prompts for ticker and interval)
python gold_scalping_example.py

# Or use programmatically
from gold_scalping_example import gold_scalping

# Analyze GLD with 5-minute candles
signals = gold_scalping(ticker="GLD", interval="5m", period="5d")

# Analyze gold futures with 15-minute candles
signals = gold_scalping(ticker="GC=F", interval="15m", period="1mo")
```

**Supported Gold Tickers:**
- `GLD`: Gold ETF (recommended, most reliable)
- `GC=F`: Gold Futures (may have market hours restrictions)

**Output Includes:**
- 🔥 **Strong Demand & Supply Zones** with strength ratings (VERY STRONG, STRONG, MODERATE, WEAK)
- 💰 **Best Buy Price** - Optimal entry point based on strongest demand zone
- 💰 **Best Sell Price** - Optimal exit point based on strongest supply zone
- 📊 **Scalping Risk/Reward** - Entry, stop-loss, take-profit, and R:R ratio
- ⚡ **Quick Scalping Signal** - Immediate BUY/WAIT recommendation

**Option 3: Use through the full scanner**
The `get_current_signals()` function is used internally by the full scanner:
```bash
python stock_scanner.py
# or
python -m scanners.agentic_scanner
```

#### Complete Example: Individual Components

```python
from core.data import get_tech_stocks, get_stock_info
from core.analysis import calculate_all_indicators, get_current_signals, StockAIAnalyzer
from core.screening import StockScreener

# Fetch stock data
tech_stocks = get_tech_stocks(period="3mo")

# Analyze a specific stock
ticker = 'AAPL'
df = tech_stocks[ticker]
df_indicators = calculate_all_indicators(df)
signals = get_current_signals(df_indicators)

# Get stock info
info = get_stock_info(ticker)

# AI Analysis
analyzer = StockAIAnalyzer()
analysis = analyzer.analyze_stock(
    ticker=ticker,
    stock_info=info,
    technical_signals=signals,
    price_data_summary={}
)

print(f"Recommendation: {analysis['recommendation']}")
print(f"Confidence: {analysis['confidence']}%")
```

## 📁 Project Structure

The project is organized into a clean modular structure for easy expansion and maintenance:

```
ai-stock-scanner/
├── stock_scanner.py        # ⭐ SINGLE COMBINED FILE (Recommended for quick use)
│                           #    Contains all functionality in one standalone file
│                           #    No imports needed - perfect for simple execution
│
├── scanners/               # Scanner implementations and orchestrators
│   ├── __init__.py        # Package initialization - exports AgenticStockScanner
│   └── agentic_scanner.py # Unified orchestrator class
│                           #    - Supports both sequential and parallel processing
│                           #    - Initializes AI analyzer and screener
│                           #    - Manages the scanning workflow
│                           #    - Set parallel=True for faster multi-stock analysis
│                           #    - Provides CLI interface with --parallel flag
│
├── core/                   # Core modules - reusable components
│   │
│   ├── data/               # Data fetching and retrieval module
│   │   ├── __init__.py    # Package initialization - exports data functions
│   │   └── stock_fetcher.py
│   │                       #    Functions:
│   │                       #    - fetch_stock_data(): Get OHLCV data for a ticker
│   │                       #    - fetch_multiple_stocks(): Batch fetch with rate limiting
│   │                       #    - get_tech_stocks(): Fetch predefined tech stock list
│   │                       #    - get_rising_stocks(): Fetch predefined rising stocks
│   │                       #    - get_all_stocks(): Fetch all tracked stocks
│   │                       #    - get_stock_info(): Get company fundamentals
│   │                       #    Constants: TECH_STOCKS, RISING_STOCKS lists
│   │
│   ├── analysis/           # Analysis modules - technical and AI analysis
│   │   ├── __init__.py    # Package initialization - exports analysis functions/classes
│   │   │
│   │   ├── technical_analyzer.py
│   │   │                   #    Technical indicator calculations:
│   │   │                   #    - calculate_rsi(): Relative Strength Index
│   │   │                   #    - calculate_ema()/sma(): Moving averages
│   │   │                   #    - calculate_macd(): MACD indicator
│   │   │                   #    - calculate_bollinger_bands(): Bollinger Bands
│   │   │                   #    - calculate_atr(): Average True Range
│   │   │                   #    - calculate_momentum(): Momentum indicator
│   │   │                   #    - calculate_price_change(): Price changes over periods
│   │   │                   #    - calculate_volume_indicators(): Volume analysis
│   │   │                   #    - analyze_trend(): Trend direction and strength
│   │   │                   #    - calculate_all_indicators(): Compute all indicators
│   │   │                   #    - get_current_signals(): Extract current trading signals
│   │   │
│   │   └── ai_analyzer.py
│   │                       #    AI-powered stock analysis:
│   │                       #    - StockAIAnalyzer class: Main AI analyzer
│   │                       #      * analyze_stock(): Generate buy/sell/wait recommendation
│   │                       #      * Uses OpenAI GPT models for context-aware analysis
│   │                       #      * Fallback to technical analysis if AI unavailable
│   │                       #      * Returns confidence scores, reasoning, risk assessment
│   │                       #    - batch_analyze(): Analyze multiple stocks
│   │
│   └── screening/          # Screening and filtering module
│       ├── __init__.py    # Package initialization - exports StockScreener
│       └── stock_screener.py
│                           #    StockScreener class with screening methods:
│                           #    - screen_tech_stocks(): Filter by technology sector
│                           #    - screen_rising_stocks(): Filter by price momentum
│                           #    - screen_momentum_stocks(): Filter by strong momentum
│                           #    - screen_oversold_stocks(): Find oversold opportunities
│                           #    - screen_breakout_stocks(): Find breakout patterns
│                           #    - screen_by_volume(): Filter by unusual volume
│                           #    - screen_buy_opportunities(): Filter by buy signals
│                           #    - comprehensive_screen(): Multi-criteria screening
│                           #    - add_filter(): Add custom filter functions
│
├── examples/               # Example scripts and usage demonstrations
│   ├── __init__.py        # Package initialization
│   └── usage_examples.py  # Comprehensive examples
│                           #    - Get trading signals for single stock
│                           #    - Full scanner (sequential)
│                           #    - Full scanner (parallel)
│                           #    - Custom analysis template
│                           #    Run: python examples/usage_examples.py [mode]
│
├── gold_scalping_example.py  # Gold scalping with strong zone analysis
│                              #    - Intraday scalping (5m, 15m, 30m intervals)
│                              #    - Strong demand/supply zone identification
│                              #    - Best buy/sell price recommendations
│                              #    - Risk/reward calculations
│                              #    Run: python gold_scalping_example.py
│
├── utils/                  # Utility functions and helpers
│   ├── __init__.py        # Package initialization - exports utility functions
│   └── format_signals.py  # Helper functions for formatting trading signals output
│                           #    - format_signals(): Format signals dict to string
│                           #    - print_signals(): Print formatted signals directly
│
├── requirements.txt        # Python dependencies:
│                           #    - pandas: Data manipulation
│                           #    - numpy: Numerical calculations
│                           #    - yfinance: Stock data fetching
│                           #    - openai: AI analysis integration
│
└── README.md              # This documentation file
```

### Folder Organization Explained

#### 📂 `scanners/` - Scanner Implementations
Contains high-level orchestrator classes that coordinate the entire scanning workflow. These are the main entry points for running scans.

**Purpose**: Orchestrate data fetching → analysis → screening → results compilation

#### 📂 `core/data/` - Data Layer
Handles all external data retrieval from Yahoo Finance API. This is where stock price data and company information is fetched.

**Purpose**: Interface with external data sources (Yahoo Finance)

#### 📂 `core/analysis/` - Analysis Layer
Contains all analysis engines - both rule-based (technical indicators) and AI-powered analysis. This is where stock data is transformed into actionable insights.

**Purpose**: Transform raw data into trading signals and recommendations

#### 📂 `core/screening/` - Filtering Layer
Filters and screens stocks based on various criteria. Works with analysis results to identify specific opportunities.

**Purpose**: Apply filters and criteria to narrow down investment opportunities

### Which File Should I Use?

- **`stock_scanner.py`** - Single combined file with everything. **Use this if you want simplicity!**
  - ✅ All code in one file
  - ✅ Easy to run: `python stock_scanner.py`
  - ✅ No import issues
  - ✅ Perfect for quick execution

- **Separate files** (modular structure in `core/` and `scanners/`) - Modular structure
  - ✅ Better for development/customization
  - ✅ Import individual components
  - ✅ Easier to modify specific parts
  - ✅ Organized into logical subfolders for expansion
  - ⚠️ Requires Python package structure (uses imports)

## 🔧 Components - Detailed Breakdown

### 1. Data Layer (`core/data/stock_fetcher.py`)

**Location**: `core/data/stock_fetcher.py`

**Purpose**: Fetches and retrieves stock market data from Yahoo Finance API.

**Key Functions**:
- `fetch_stock_data(ticker, period, interval)` - Retrieves OHLCV (Open/High/Low/Close/Volume) data for a single stock
- `fetch_multiple_stocks(tickers, period, interval, delay)` - Batch fetches data for multiple stocks with rate limiting
- `get_tech_stocks(period, interval)` - Fetches data for predefined tech stock list (AAPL, MSFT, GOOGL, NVDA, etc.)
- `get_rising_stocks(period, interval)` - Fetches data for predefined rising stocks (SMCI, ARM, RDDT, etc.)
- `get_all_stocks(period, interval)` - Fetches data for all tracked stocks (tech + rising)
- `get_stock_info(ticker)` - Retrieves company fundamentals (sector, P/E ratio, market cap, 52-week highs/lows, etc.)

**Constants**:
- `TECH_STOCKS` - List of technology stock tickers (~35 stocks)
- `RISING_STOCKS` - List of rising/momentum stock tickers (~14 stocks)

**Dependencies**: `yfinance`, `pandas`

---

### 2. Technical Analysis (`core/analysis/technical_analyzer.py`)

**Location**: `core/analysis/technical_analyzer.py`

**Purpose**: Calculates technical indicators and generates trading signals from price data.

**Key Functions**:

**Indicator Calculations**:
- `calculate_rsi(series, period=14)` - Relative Strength Index (identifies overbought/oversold)
- `calculate_ema(series, period)` - Exponential Moving Average
- `calculate_sma(series, period)` - Simple Moving Average
- `calculate_macd(series, fast=12, slow=26, signal=9)` - MACD indicator with signal line
- `calculate_bollinger_bands(series, period=20, std_dev=2)` - Bollinger Bands (volatility indicator)
- `calculate_atr(df, period=14)` - Average True Range (volatility measure)
- `calculate_momentum(series, period=10)` - Momentum indicator
- `calculate_price_change(df, periods=[1,5,10,20])` - Price changes over multiple timeframes
- `calculate_volume_indicators(df)` - Volume-based indicators (volume SMA, volume ratio)

**Analysis Functions**:
- `analyze_trend(df)` - Determines trend direction (up/down/sideways) and strength (0-100)
- `identify_demand_supply_zones(df, lookback_period=50)` - Identifies demand (support) and supply (resistance) zones:
  - Finds swing highs/lows where price bounced significantly
  - Returns nearest zones above/below current price with distances
  - Provides list of key support/resistance levels
  
- `generate_buy_recommendation(df, signals)` - Generates buy recommendation with entry price:
  - Analyzes RSI, trend, EMA, MACD, and demand zones to score opportunities (0-100)
  - Suggests entry price based on demand zones, oversold conditions, or pullbacks
  - Calculates stop-loss (3% below demand zone or entry)
  - Calculates take-profit (near supply zone or 8% target)
  - Provides risk/reward ratio
  
- `calculate_all_indicators(df)` - Computes all technical indicators and adds to DataFrame
  
- `get_current_signals(df)` - Extracts current trading signals from the most recent data:
  - RSI signal (oversold/overbought/neutral)
  - EMA cross (bullish/bearish)
  - MACD signal (bullish/bearish)
  - Bollinger position (upper/middle/lower)
  - Price changes (1d, 5d, 20d)
  - Trend direction and strength
  - **Demand and supply zones** (nearest zones, distances, full lists)
  - **Buy recommendation** (BUY/CONSIDER BUY/WATCH/WAIT) with score
  - **Suggested entry price** with strategy explanation
  - **Stop-loss and take-profit levels**
  - **Risk/reward ratio**

**Dependencies**: `pandas`, `numpy`

---

### 3. AI Analysis (`core/analysis/ai_analyzer.py`)

**Location**: `core/analysis/ai_analyzer.py`

**Purpose**: Uses AI (OpenAI GPT models) to provide context-aware stock analysis and recommendations.

**Key Class**: `StockAIAnalyzer`

**Methods**:
- `__init__(api_key, model="gpt-4o-mini")` - Initialize with OpenAI API key (optional)
- `analyze_stock(ticker, stock_info, technical_signals, price_data_summary)` - Main analysis method:
  - Prepares context from technical indicators and fundamentals
  - Sends to OpenAI API for analysis
  - Parses AI response into structured recommendation
  - Returns: recommendation (BUY/SELL/WAIT), confidence (0-100), reasoning, upside_potential, risk_level
- `batch_analyze(stocks_data)` - Analyze multiple stocks in batch

**Features**:
- Context-aware analysis combining technical indicators, fundamentals, and market conditions
- Fallback to rule-based technical analysis if AI unavailable
- Configurable model (gpt-4o-mini, gpt-4, gpt-3.5-turbo)
- Automatic technical score calculation as backup

**Dependencies**: `openai` (optional - has fallback)

---

### 4. Screening (`core/screening/stock_screener.py`)

**Location**: `core/screening/stock_screener.py`

**Purpose**: Filters and screens stocks based on various criteria to identify opportunities.

**Key Class**: `StockScreener`

**Screening Methods**:
- `screen_tech_stocks(stocks_data, stock_infos)` - Filters stocks by technology sector
- `screen_rising_stocks(stocks_data, min_change_5d=2.0, min_change_20d=5.0)` - Finds stocks with upward price momentum
- `screen_momentum_stocks(stocks_data, min_momentum=5.0)` - Identifies stocks with strong momentum
- `screen_oversold_stocks(stocks_data, max_rsi=35)` - Finds oversold stocks (potential reversal opportunities)
- `screen_breakout_stocks(stocks_data)` - Identifies stocks breaking above Bollinger Bands (breakout patterns)
- `screen_by_volume(stocks_data, min_volume_ratio=1.5)` - Finds stocks with unusually high volume
- `screen_buy_opportunities(stocks_data, stock_infos, analysis_results)` - Filters stocks with BUY recommendations and confidence ≥60%
- `comprehensive_screen(stocks_data, stock_infos, analysis_results, criteria)` - Multi-criteria screening:
  - Filter by tech_only, min_confidence, min_upside, max_risk, rising_only
- `add_filter(filter_func)` - Add custom filter functions

**Dependencies**: `pandas`, imports from `core.analysis` for technical indicators

---

### 5. Scanner Orchestrator (`scanners/agentic_scanner.py`)

**Location**: `scanners/agentic_scanner.py`

**Purpose**: Main orchestrator that coordinates the entire scanning workflow.

**Key Class**: `AgenticStockScanner`

**Initialization**:
- `__init__(openai_api_key, model="gpt-4o-mini")` - Creates AI analyzer and screener instances

**Main Methods**:
- `scan_stocks(tickers=None, period="3mo", interval="1d", focus="all")` - Main scanning workflow:
  1. Fetches stock data (tech, rising, or all stocks)
  2. Gathers stock information (fundamentals)
  3. Calculates technical indicators
  4. Runs AI analysis on each stock
  5. Screens for buy opportunities
  6. Compiles and returns results
- `print_recommendations(limit=10)` - Pretty-prints top recommendations
- `save_results(filename=None)` - Saves scan results to JSON file
- `_compile_recommendations()` - Internal method to format recommendations

**CLI Interface**:
- Supports command-line arguments: `--focus`, `--period`, `--api-key`, `--limit`, `--save`
- Can be run as: `python -m scanners.agentic_scanner [options]`

**Dependencies**: All core modules (data, analysis, screening)

---

### 6. Standalone Scanner (`stock_scanner.py`)

**Location**: Root directory (`stock_scanner.py`)

**Purpose**: Self-contained single-file version with all functionality embedded.

**Features**:
- No external imports (except libraries)
- All code in one file (~1060 lines)
- Same functionality as modular version
- Easier for quick execution
- No import path issues

**Usage**: `python stock_scanner.py [options]`

**Best For**: Quick scans, simple deployments, users who want everything in one file

---

## 🔄 Data Flow & Module Interactions

Understanding how the modules work together:

```
┌─────────────────────────────────────────────────────────────┐
│                    Scanner Workflow                          │
└─────────────────────────────────────────────────────────────┘

1. DATA FETCHING (core/data/stock_fetcher.py)
   ├── Fetch stock price data (OHLCV) from Yahoo Finance
   ├── Get company fundamentals (sector, P/E, market cap)
   └── Return: Raw stock data DataFrames

2. TECHNICAL ANALYSIS (core/analysis/technical_analyzer.py)
   ├── Input: Raw price data
   ├── Calculate indicators (RSI, MACD, EMA, Bollinger Bands, etc.)
   ├── Analyze trends and momentum
   └── Return: Technical signals dictionary

3. AI ANALYSIS (core/analysis/ai_analyzer.py)
   ├── Input: Technical signals + fundamentals
   ├── Send context to OpenAI GPT model
   ├── Parse AI recommendation
   └── Return: BUY/SELL/WAIT with confidence score

4. SCREENING (core/screening/stock_screener.py)
   ├── Input: Analysis results + stock data
   ├── Apply filters (tech, momentum, oversold, etc.)
   └── Return: List of filtered stock tickers

5. ORCHESTRATION (scanners/agentic_scanner.py)
   ├── Coordinates steps 1-4
   ├── Compiles results
   ├── Formats output
   └── Returns: Complete scan results with recommendations
```

### Module Dependencies

```
stock_scanner.py (standalone - no dependencies on other modules)
    │
    └── All code embedded

scanners/agentic_scanner.py
    ├── depends on → core/data/stock_fetcher.py
    ├── depends on → core/analysis/technical_analyzer.py
    ├── depends on → core/analysis/ai_analyzer.py
    └── depends on → core/screening/stock_screener.py

core/screening/stock_screener.py
    └── depends on → core/analysis/technical_analyzer.py

core/analysis/ai_analyzer.py
    └── optional dependency → openai library

core/analysis/technical_analyzer.py
    └── dependencies → pandas, numpy

core/data/stock_fetcher.py
    └── dependencies → yfinance, pandas
```
---

## 📈 Output Format

The scanner returns a dictionary with:

```python
{
    'scan_timestamp': '2025-12-08T12:00:00',
    'stocks_scanned': 50,
    'buy_opportunities': 12,
    'high_confidence_buys': 5,
    'recommendations': [
        {
            'ticker': 'AAPL',
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'current_price': 180.50,
            'recommendation': 'BUY',
            'confidence': 85,
            'upside_potential': 'High',
            'risk_level': 'Low',
            'reasoning': 'Strong upward trend with bullish technical indicators...',
            'technical_score': 82.5,
            'price_change_1d': 1.2,
            'price_change_5d': 3.5,
            'price_change_20d': 8.2,
            'trend': 'up',
            'rsi': 45.3
        },
        ...
    ],
    'all_analysis': {...}  # Full analysis for all stocks
}
```

## ⚙️ Configuration

### Focus Areas

- `"all"`: Scan all tracked stocks (tech + rising)
- `"tech"`: Focus on technology stocks only
- `"rising"`: Focus on rising momentum stocks only

### Time Periods

- `"1mo"`: 1 month of data
- `"3mo"`: 3 months of data (recommended)
- `"6mo"`: 6 months of data
- `"1y"`: 1 year of data

### Intervals

- `"1d"`: Daily data (recommended for swing trading)
- `"1h"`: Hourly data (for day trading)
- `"5m"`: 5-minute data (for scalping)
- `"15m"`: 15-minute data (for short-term scalping)
- `"30m"`: 30-minute data (for medium scalping)

**Note**: For scalping with intraday intervals, see `gold_scalping_example.py` which provides specialized zone strength analysis and optimal buy/sell price identification.

## 🎯 Recommendation Criteria

The system recommends BUY when:
- AI confidence ≥ 60% (high confidence ≥ 70%)
- Technical score indicates bullish conditions
- Trend is upward with strong momentum
- Risk/reward ratio is favorable

Recommendations include:
- **Confidence Score**: 0-100% (higher = more confident)
- **Upside Potential**: Low/Medium/High
- **Risk Level**: Low/Medium/High
- **Reasoning**: AI-generated explanation

## ⚠️ Important Notes

1. **Not Financial Advice**: This tool is for educational and research purposes only. Always do your own research.

2. **API Costs**: Using OpenAI API incurs costs. Monitor your usage. The system works without it (using technical analysis only).

3. **Rate Limiting**: Built-in delays prevent API rate limits. Scanning many stocks takes time.

4. **Data Accuracy**: Data comes from Yahoo Finance. Verify critical information before trading.

5. **Market Conditions**: Recommendations are based on technical analysis and AI reasoning. Market conditions can change rapidly.

6. **Risk Management**: Always use stop-losses and proper position sizing. Never risk more than you can afford to lose.

7. **404 Errors**: Some stocks (like ANSS, SPLK) may show 404 errors if they're delisted or data is unavailable. These have been removed from the default stock lists.

## 📝 License

This project is provided as-is for educational and research purposes.

---

**Need Help?** Check the code comments or open an issue.


