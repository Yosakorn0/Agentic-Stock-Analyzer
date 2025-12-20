# Data Sources Setup Guide

This guide explains how to set up different data sources for fetching XAUUSD (Gold) price data, especially for accurate TradingView-compatible data.

## 🎯 Recommended Data Sources (Best to Good)

### 1. **TradingView** ⭐ (Recommended - Most Accurate)

TradingView provides the most accurate data that matches what you see on TradingView charts.

#### Setup:
```bash
pip install pytradingview
```

#### Usage:
The scripts will automatically use TradingView if `pytradingview` is installed. No API key needed!

#### Advantages:
- ✅ Free (no API key required)
- ✅ Matches TradingView charts exactly
- ✅ Real-time and historical data
- ✅ Works for XAUUSD and other forex pairs

#### Limitations:
- ⚠️ Rate limits may apply with heavy usage
- ⚠️ Requires internet connection

---

### 2. **OANDA API** ⭐ (Professional Forex Data)

OANDA provides professional-grade forex and commodity data with a free tier.

#### Setup:
1. Sign up for free at: https://www.oanda.com/us-en/trading/api/
2. Get your API key from the OANDA dashboard
3. Set environment variable:
   ```bash
   # Windows PowerShell
   $env:OANDA_API_KEY="your_api_key_here"
   
   # Windows CMD
   set OANDA_API_KEY=your_api_key_here
   
   # Linux/Mac
   export OANDA_API_KEY=your_api_key_here
   ```

#### Advantages:
- ✅ Free tier available
- ✅ Professional-grade data
- ✅ Reliable and fast
- ✅ Good for automated trading

#### Limitations:
- ⚠️ Requires account signup
- ⚠️ Free tier has rate limits

---

### 3. **Yahoo Finance** (Fallback)

Yahoo Finance is used as a fallback when other sources are unavailable.

#### Setup:
Already included via `yfinance` package (installed by default).

#### Usage:
The scripts automatically fall back to Yahoo Finance using `GC=F` (Gold Futures) when other sources fail.

#### Advantages:
- ✅ No setup required
- ✅ Free
- ✅ Works for many assets

#### Limitations:
- ⚠️ XAUUSD intraday data not available
- ⚠️ Uses GC=F (futures) instead of spot XAUUSD
- ⚠️ May have data gaps or delays
- ⚠️ Limited to ~60 days for 5-minute intervals

---

## 📊 Data Source Priority

The scripts automatically try sources in this order:

1. **TradingView** (if `pytradingview` installed)
2. **OANDA** (if `OANDA_API_KEY` environment variable set)
3. **Yahoo Finance** (fallback to GC=F)

---

## 🔧 Quick Setup for Best Results

### Option A: TradingView (Easiest - Recommended)
```bash
pip install pytradingview
```
That's it! The scripts will automatically use TradingView.

### Option B: OANDA (Most Reliable)
```bash
# 1. Sign up at https://www.oanda.com/us-en/trading/api/
# 2. Get your API key
# 3. Set environment variable
export OANDA_API_KEY="your_key_here"  # Linux/Mac
# or
$env:OANDA_API_KEY="your_key_here"     # Windows PowerShell
```

### Option C: Use Both (Best Coverage)
```bash
pip install pytradingview
export OANDA_API_KEY="your_key_here"
```
The script will try TradingView first, then OANDA if TradingView fails.

---

## 🧪 Testing Your Setup

Test which data sources are available:

```bash
python -c "from core.data.forex_fetcher import fetch_xauusd_data; df = fetch_xauusd_data(period='5d', interval='5m'); print('✅ Success!' if df is not None else '❌ Failed')"
```

Or test individual sources:

```python
from core.data.forex_fetcher import fetch_tradingview_xauusd, fetch_oanda_xauusd

# Test TradingView
df = fetch_tradingview_xauusd(period="5d", interval="5m")
print("TradingView:", "✅" if df is not None else "❌")

# Test OANDA
df = fetch_oanda_xauusd(period="5d", interval="5m")
print("OANDA:", "✅" if df is not None else "❌")
```

---

## 📝 Environment Variables

Create a `.env` file in the project root (optional):

```env
# OANDA API (optional)
OANDA_API_KEY=your_oanda_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here

# Other API keys (if you add more sources later)
# TWELVE_DATA_API_KEY=your_key
# ALPHA_VANTAGE_API_KEY=your_key
```

The scripts will automatically load these via `python-dotenv`.

---

## 🚀 Running Gold Scripts

Once set up, run the gold scripts as usual:

```bash
# Backtest
python gold_scalping_backtest.py --period 60d --interval 5m

# Live monitoring
python gold_scalping_live.py

# Analysis
python gold_scalping.py
```

The scripts will automatically use the best available data source!

---

## ❓ Troubleshooting

### "All sources failed"
- **Check internet connection**
- **Install TradingView**: `pip install pytradingview`
- **Or set up OANDA**: Get API key and set `OANDA_API_KEY` environment variable
- **Check Yahoo Finance fallback**: Should work automatically

### "TradingView timeout"
- TradingView may be rate-limiting
- Try OANDA as alternative
- Or wait a few minutes and retry

### "OANDA authentication failed"
- Verify your API key is correct
- Check that `OANDA_API_KEY` environment variable is set
- Make sure you're using a practice account API key (not live trading)

### "No data returned"
- Check that the period/interval combination is valid
- For 5m intervals, use period <= 60d
- For 1d intervals, you can use longer periods (1y, 5y, etc.)

---

## 📚 Additional Resources

- **TradingView**: https://www.tradingview.com/
- **OANDA API Docs**: https://developer.oanda.com/
- **Yahoo Finance**: https://finance.yahoo.com/

---

## 💡 Recommendations

**For best accuracy (matches TradingView charts):**
- Use TradingView via `pytradingview` ⭐

**For production/automated trading:**
- Use OANDA API (more reliable, professional-grade)

**For quick testing:**
- Yahoo Finance fallback works fine (uses GC=F)

**For maximum reliability:**
- Set up both TradingView and OANDA (automatic fallback)

