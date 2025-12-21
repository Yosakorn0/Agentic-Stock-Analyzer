"""
Enhanced Fundamentals Data Fetcher
Fetches comprehensive financial data
"""
import yfinance as yf
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


def get_enhanced_fundamentals(ticker: str, timeout: int = 30) -> Dict:
    """
    Get comprehensive fundamental data for a stock
    
    Args:
        ticker: Stock ticker symbol
        timeout: Request timeout in seconds
    
    Returns:
        Dict with all fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: stock.info)
            info = future.result(timeout=timeout)
            
            future2 = executor.submit(lambda: stock.financials)
            financials = future2.result(timeout=timeout)
            
            future3 = executor.submit(lambda: stock.balance_sheet)
            balance_sheet = future3.result(timeout=timeout)
            
            future4 = executor.submit(lambda: stock.cashflow)
            cashflow = future4.result(timeout=timeout)
        
        # Extract key metrics
        revenue = None
        revenue_growth = None
        if 'Total Revenue' in financials.index and len(financials.loc['Total Revenue']) > 0:
            revenue = financials.loc['Total Revenue'].iloc[0]
            revenue_growth = _calculate_growth(financials, 'Total Revenue')
        
        net_income = None
        earnings_growth = None
        if 'Net Income' in financials.index and len(financials.loc['Net Income']) > 0:
            net_income = financials.loc['Net Income'].iloc[0]
            earnings_growth = _calculate_growth(financials, 'Net Income')
        
        # Balance sheet metrics
        total_debt = None
        total_equity = None
        debt_to_equity = None
        
        if 'Total Debt' in balance_sheet.index and len(balance_sheet.loc['Total Debt']) > 0:
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
        if 'Stockholders Equity' in balance_sheet.index and len(balance_sheet.loc['Stockholders Equity']) > 0:
            total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
        
        if total_debt and total_equity and total_equity != 0:
            debt_to_equity = (total_debt / total_equity) * 100
        
        # Cash flow
        operating_cashflow = None
        if 'Operating Cash Flow' in cashflow.index and len(cashflow.loc['Operating Cash Flow']) > 0:
            operating_cashflow = cashflow.loc['Operating Cash Flow'].iloc[0]
        
        # Calculate ratios
        market_cap = info.get('marketCap', 0)
        shares_outstanding = info.get('sharesOutstanding', 0)
        book_value = info.get('bookValue', None)
        roe = info.get('returnOnEquity', None)
        
        return {
            # Basic info
            'market_cap': market_cap,
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'peg_ratio': info.get('pegRatio', None),
            
            # Revenue & Earnings
            'revenue': revenue,
            'revenue_growth_yoy': revenue_growth,
            'net_income': net_income,
            'earnings_growth_yoy': earnings_growth,
            'eps': info.get('trailingEps', None),
            'eps_growth': None,  # Calculate from historical if needed
            
            # Profitability
            'profit_margin': info.get('profitMargins', None),
            'operating_margin': info.get('operatingMargins', None),
            'gross_margin': info.get('grossMargins', None),
            
            # Financial Health
            'debt_to_equity': debt_to_equity,
            'current_ratio': info.get('currentRatio', None),
            'quick_ratio': info.get('quickRatio', None),
            'total_debt': total_debt,
            'total_cash': info.get('totalCash', None),
            'operating_cashflow': operating_cashflow,
            
            # Returns
            'roe': roe,
            'roa': info.get('returnOnAssets', None),
            'roic': info.get('returnOnInvestedCapital', None),
            
            # Valuation
            'book_value': book_value,
            'price_to_book': info.get('priceToBook', None),
            'price_to_sales': info.get('priceToSalesTrailing12Months', None),
            'enterprise_value': info.get('enterpriseValue', None),
            
            # Dividends
            'dividend_yield': info.get('dividendYield', 0),
            'payout_ratio': info.get('payoutRatio', None),
            
            # Analyst data
            'target_price': info.get('targetMeanPrice', None),
            'recommendation': info.get('recommendationKey', None),
            'number_of_analysts': info.get('numberOfAnalystOpinions', None),
            
            # Additional context
            '52_week_high': info.get('fiftyTwoWeekHigh', None),
            '52_week_low': info.get('fiftyTwoWeekLow', None),
            'beta': info.get('beta', None),
            'volume': info.get('volume', None),
            'avg_volume': info.get('averageVolume', None)
        }
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {str(e)[:100]}")
        return {}


def _calculate_growth(financials, metric: str) -> Optional[float]:
    """Calculate year-over-year growth for a metric"""
    try:
        if metric not in financials.index:
            return None
        
        values = financials.loc[metric]
        if len(values) < 2:
            return None
        
        current = values.iloc[0]
        previous = values.iloc[1]
        
        if previous == 0 or previous is None or current is None:
            return None
        
        growth = ((current - previous) / abs(previous)) * 100
        return growth
    except Exception:
        return None

