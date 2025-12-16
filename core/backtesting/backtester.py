"""
Backtesting Engine - Validates strategy performance on historical data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


class Backtester:
    """
    Backtesting engine for validating trading strategies
    """
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.positions = {}
        self.equity_curve = []
    
    def backtest_strategy(self, 
                         historical_data: Dict[str, pd.DataFrame],
                         signals: Dict[str, Dict],
                         analyzer_type: str = "technical") -> Dict:
        """
        Backtest a trading strategy on historical data
        
        Args:
            historical_data: Dict of ticker -> historical price data
            signals: Dict of ticker -> trading signals (BUY/SELL/WAIT) by date
            analyzer_type: Type of analyzer ('technical', 'ai', 'ml')
            
        Returns:
            Dictionary with backtest results
        """
        print("🔄 Running backtest...")
        
        capital = self.initial_capital
        positions = {}  # ticker -> {'shares': int, 'entry_price': float, 'entry_date': date}
        trades = []
        equity_curve = [{'date': None, 'capital': capital, 'positions_value': 0, 'total_equity': capital}]
        
        # Sort dates across all stocks
        all_dates = set()
        for df in historical_data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)
        
        if not all_dates:
            print("❌ No historical data available")
            return self._empty_results()
        
        for date in all_dates:
            daily_pnl = 0
            positions_value = 0
            
            # Update existing positions
            for ticker, position in list(positions.items()):
                if ticker in historical_data and date in historical_data[ticker].index:
                    current_price = historical_data[ticker].loc[date, 'close']
                    position_value = position['shares'] * current_price
                    positions_value += position_value
                    
                    # Check exit conditions
                    if ticker in signals and date in signals[ticker]:
                        signal = signals[ticker][date]
                        if signal.get('recommendation') == 'SELL':
                            # Exit position
                            exit_value = position['shares'] * current_price
                            commission_cost = exit_value * self.commission
                            pnl = exit_value - position['entry_value'] - commission_cost
                            
                            trades.append({
                                'ticker': ticker,
                                'entry_date': str(position['entry_date']),
                                'exit_date': str(date),
                                'entry_price': float(position['entry_price']),
                                'exit_price': float(current_price),
                                'shares': int(position['shares']),
                                'pnl': float(pnl),
                                'return_pct': float((pnl / position['entry_value']) * 100)
                            })
                            
                            capital += exit_value - commission_cost
                            del positions[ticker]
            
            # Check for new entry signals
            for ticker, df in historical_data.items():
                if date not in df.index:
                    continue
                
                if ticker in signals and date in signals[ticker]:
                    signal = signals[ticker][date]
                    recommendation = signal.get('recommendation', 'WAIT')
                    confidence = signal.get('confidence', 0)
                    
                    if recommendation in ['BUY', 'CONSIDER BUY'] and confidence >= 60:
                        if ticker not in positions:
                            current_price = df.loc[date, 'close']
                            
                            # Position sizing (risk 2% of capital per trade)
                            risk_amount = capital * 0.02
                            stop_loss_pct = 0.03  # 3% stop loss
                            shares = int((risk_amount / (current_price * stop_loss_pct)) / current_price)
                            
                            if shares > 0 and capital >= shares * current_price:
                                entry_value = shares * current_price
                                commission_cost = entry_value * self.commission
                                
                                positions[ticker] = {
                                    'shares': shares,
                                    'entry_price': current_price,
                                    'entry_date': date,
                                    'entry_value': entry_value
                                }
                                
                                capital -= (entry_value + commission_cost)
            
            # Record equity
            total_equity = capital + positions_value
            equity_curve.append({
                'date': str(date),
                'capital': float(capital),
                'positions_value': float(positions_value),
                'total_equity': float(total_equity)
            })
        
        # Close remaining positions at last date
        last_date = all_dates[-1]
        for ticker, position in positions.items():
            if ticker in historical_data and last_date in historical_data[ticker].index:
                exit_price = historical_data[ticker].loc[last_date, 'close']
                exit_value = position['shares'] * exit_price
                commission_cost = exit_value * self.commission
                pnl = exit_value - position['entry_value'] - commission_cost
                
                trades.append({
                    'ticker': ticker,
                    'entry_date': str(position['entry_date']),
                    'exit_date': str(last_date),
                    'entry_price': float(position['entry_price']),
                    'exit_price': float(exit_price),
                    'shares': int(position['shares']),
                    'pnl': float(pnl),
                    'return_pct': float((pnl / position['entry_value']) * 100)
                })
                
                capital += exit_value - commission_cost
        
        # Calculate metrics
        final_capital = capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = (len(winning_trades) / len(trades)) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            total_win = sum([t['pnl'] for t in winning_trades])
            total_loss = abs(sum([t['pnl'] for t in losing_trades]))
            profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
            
            # Sharpe ratio (simplified)
            returns = [t['return_pct'] for t in trades]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) if returns else 0
            
            # Max drawdown
            equity_values = [e['total_equity'] for e in equity_curve if e['total_equity']]
            if equity_values:
                peak = equity_values[0]
                max_drawdown = 0
                for value in equity_values:
                    if value > peak:
                        peak = value
                    drawdown = ((peak - value) / peak) * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            else:
                max_drawdown = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        results = {
            'initial_capital': float(self.initial_capital),
            'final_capital': float(final_capital),
            'total_return': float(total_return),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades) if trades else 0,
            'losing_trades': len(losing_trades) if trades else 0,
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'trades': trades,
            'equity_curve': equity_curve,
            'analyzer_type': analyzer_type
        }
        
        print(f"✅ Backtest complete!")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Capital: ${final_capital:,.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Total Trades: {len(trades)}")
        if trades:
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Profit Factor: {profit_factor:.2f}")
            print(f"   Max Drawdown: {max_drawdown:.2f}%")
        
        return results
    
    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'initial_capital': float(self.initial_capital),
            'final_capital': float(self.initial_capital),
            'total_return': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'trades': [],
            'equity_curve': [],
            'analyzer_type': 'unknown'
        }
    
    def save_results(self, results: Dict, filepath: str):
        """Save backtest results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {filepath}")

