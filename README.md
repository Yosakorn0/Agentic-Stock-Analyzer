# Agentic AI Stock Scanner

An intelligent AI-powered system that scans tech stocks and rising stocks to identify high-potential buy opportunities RIGHT NOW.

## 🎯 Features

- **AI-Powered Analysis**: Uses OpenAI GPT models to analyze stocks with context-aware reasoning
- **Technical Analysis**: Comprehensive technical indicators (RSI, MACD, EMA, Bollinger Bands, etc.)
- **Smart Screening**: Filters stocks by tech sector, rising momentum, oversold conditions, and more
- **Real-Time Scanning**: Scans multiple stocks simultaneously with rate limiting
- **Buy Recommendations**: Generates actionable buy/sell/wait recommendations with confidence scores
- **Risk Assessment**: Evaluates upside potential and risk levels for each opportunity

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

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

# Show top 20 recommendations
python stock_scanner.py --limit 20

# Save results to JSON file
python stock_scanner.py --save
```

#### Option B: Modular Files (For Custom Development)

**Standard Version (Sequential Processing):**
```bash
# Scan all stocks (tech + rising)
python -m scanners.agentic_scanner

# Focus on tech stocks only
python -m scanners.agentic_scanner --focus tech

# Focus on rising stocks only
python -m scanners.agentic_scanner --focus rising

# Show top 20 recommendations
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

**Or use CLI with parallel flag:**
```bash
python -m scanners.agentic_scanner --parallel --workers 5
```

**Note:** Parallel processing significantly speeds up analysis when scanning many stocks, but uses more API rate limit quota. Use `max_workers` to control concurrency.

**Note**: `stock_scanner.py` is a single combined file with all functionality. Use it if you want everything in one place. The separate files (`agentic_scanner.py`, `ai_analyzer.py`, etc.) are for modular use.

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
- `calculate_all_indicators(df)` - Computes all technical indicators and adds to DataFrame
- `get_current_signals(df)` - Extracts current trading signals from the most recent data:
  - RSI signal (oversold/overbought/neutral)
  - EMA cross (bullish/bearish)
  - MACD signal (bullish/bearish)
  - Bollinger position (upper/middle/lower)
  - Price changes (1d, 5d, 20d)
  - Trend direction and strength

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


