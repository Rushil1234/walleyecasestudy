"""
Contrarian trading strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def make_naive(dt):
    if hasattr(dt, 'tz_localize'):
        return dt.tz_localize(None)
    elif hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


class ContrarianTrader:
    """
    Implements contrarian trading strategy based on filtered signals.
    """
    
    def __init__(self):
        """
        Initialize the contrarian trader.
        """
        self.positions = []
        self.trade_history = []
        
    def backtest_strategy(
        self,
        equity_data: Dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        config: Dict
    ) -> Dict:
        """
        Backtest the contrarian trading strategy.
        
        Args:
            equity_data: Dictionary of equity DataFrames
            signals: DataFrame with trading signals
            config: Trading configuration
            
        Returns:
            Dictionary with backtest results
        """
        if not equity_data or signals.empty:
            return self._empty_results()
        
        # Get primary asset
        primary_symbol = config.get('assets', {}).get('primary', 'XOP')
        if primary_symbol not in equity_data:
            logger.error(f"Primary asset {primary_symbol} not found")
            return self._empty_results()
        
        primary_data = equity_data[primary_symbol]
        
        # Initialize portfolio
        portfolio = self._initialize_portfolio(config)
        
        # Get benchmark data
        benchmark_symbol = 'SPY'
        benchmark_data = equity_data.get(benchmark_symbol, primary_data)
        
        # Run backtest
        results = self._run_backtest(
            primary_data=primary_data,
            signals=signals,
            portfolio=portfolio,
            benchmark_data=benchmark_data,
            config=config
        )
        
        return results
    
    def live_trading(
        self,
        equity_data: Dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        config: Dict
    ) -> Dict:
        """
        Simulate live trading (placeholder for real implementation).
        
        Args:
            equity_data: Dictionary of equity DataFrames
            signals: DataFrame with trading signals
            config: Trading configuration
            
        Returns:
            Dictionary with live trading results
        """
        # For now, just run backtest
        return self.backtest_strategy(equity_data, signals, config)
    
    def _initialize_portfolio(self, config: Dict) -> Dict:
        """
        Initialize portfolio for backtesting.
        
        Args:
            config: Trading configuration
            
        Returns:
            Portfolio dictionary
        """
        trading_config = config.get('trading', {})
        
        portfolio = {
            'cash': 100000.0,  # Starting cash
            'positions': {},   # Current positions
            'total_value': 100000.0,
            'daily_returns': [],
            'cumulative_returns': [],
            'trades': []
        }
        
        return portfolio
    
    def _run_backtest(
        self,
        primary_data: pd.DataFrame,
        signals: pd.DataFrame,
        portfolio: Dict,
        benchmark_data: pd.DataFrame,
        config: Dict
    ) -> Dict:
        """
        Run the actual backtest with proper position management.
        
        Args:
            primary_data: Primary asset data
            signals: Trading signals
            portfolio: Portfolio state
            benchmark_data: Benchmark data
            config: Trading configuration
            
        Returns:
            Backtest results
        """
        trading_config = config.get('trading', {})
        
        # Initialize tracking
        daily_values = []
        daily_returns = []
        benchmark_returns = []
        
        # Get common date range
        start_date = max(primary_data.index.min(), signals.index.min())
        end_date = min(primary_data.index.max(), signals.index.max())
        
        # Position management parameters
        max_hold_days = trading_config.get('max_hold_days', 10)  # Maximum days to hold position
        stop_loss_pct = trading_config.get('stop_loss_pct', 0.05)  # 5% stop loss
        take_profit_pct = trading_config.get('take_profit_pct', 0.10)  # 10% take profit
        
        # Run simulation day by day
        for date in pd.date_range(start=start_date, end=end_date, freq='D'):
            if date not in primary_data.index:
                continue
            
            # Get current price
            current_price = primary_data.loc[date, 'Close']
            
            # Check for position exits (stop loss, take profit, time-based)
            self._check_position_exits(portfolio, current_price, date, max_hold_days, stop_loss_pct, take_profit_pct)
            
            # Check for signals
            if date in signals.index:
                signal = signals.loc[date]
                self._execute_signal(signal, current_price, portfolio, config)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(portfolio, current_price)
            portfolio['total_value'] = portfolio_value
            
            # Calculate daily return
            if daily_values:
                daily_return = (portfolio_value - daily_values[-1]) / daily_values[-1]
            else:
                daily_return = 0.0
            
            daily_values.append(portfolio_value)
            daily_returns.append(daily_return)
            
            # Calculate benchmark return
            if date in benchmark_data.index:
                benchmark_return = benchmark_data.loc[date, 'Returns'] if 'Returns' in benchmark_data.columns else 0.0
            else:
                benchmark_return = 0.0
            benchmark_returns.append(benchmark_return)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(
            daily_values=daily_values,
            daily_returns=daily_returns,
            benchmark_returns=benchmark_returns,
            portfolio=portfolio,
            config=config
        )
        
        # Add proper date index for cumulative returns
        if daily_values:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            date_range = date_range[:len(daily_values)]  # Ensure same length
            
            # Create proper cumulative returns series with date index
            cumulative_returns = pd.Series(
                [v / daily_values[0] - 1 for v in daily_values],
                index=date_range
            )
            
            # Create benchmark cumulative returns series
            benchmark_cumulative = pd.Series(
                np.cumprod(1 + np.array(benchmark_returns)) - 1,
                index=date_range
            )
            
            # Update results with proper series
            results['cumulative_returns'] = cumulative_returns
            results['benchmark_returns'] = benchmark_cumulative
            results['daily_returns'] = pd.Series(daily_returns, index=date_range)
            results['portfolio_values'] = pd.Series(daily_values, index=date_range)
        
        return results
    
    def _execute_signal(
        self,
        signal: pd.Series,
        current_price: float,
        portfolio: Dict,
        config: Dict
    ):
        """
        Execute a trading signal with enhanced logic.
        
        Args:
            signal: Signal data
            current_price: Current asset price
            portfolio: Portfolio state
            config: Trading configuration
        """
        try:
            # Get signal type (BUY/SELL)
            signal_type = signal.get('signal_type', 'UNKNOWN')
            signal_strength = signal.get('signal_strength', 0)
            signal_quality = signal.get('signal_quality', 0)
            
            # Skip if signal is too weak or low quality
            if signal_strength < 0.1 or signal_quality < 0.3:
                return
            
            # Calculate position size based on signal strength and quality
            position_size = min(1.0, signal_strength * signal_quality * 2)  # Scale position size
            
            # Get trading parameters
            trading_config = config.get('trading', {})
            max_position_size = trading_config.get('max_position_size', 0.2)  # 20% max position
            position_size = min(position_size, max_position_size)
            
            # Calculate position value
            position_value = portfolio['cash'] * position_size
            
            # Check if we have enough cash
            if position_value > portfolio['cash']:
                position_value = portfolio['cash'] * 0.95  # Use 95% of available cash
            
            if position_value < 1000:  # Minimum position size
                return
            
            # Calculate shares
            shares = position_value / current_price
            
            # Create position
            position = {
                'id': len(portfolio['trades']) + 1,
                'type': signal_type,
                'entry_price': current_price,
                'shares': shares,
                'entry_date': datetime.now(),
                'signal_strength': signal_strength,
                'signal_quality': signal_quality,
                'position_size': position_size
            }
            
            # Add to positions
            portfolio['positions'][position['id']] = position
            
            # Update cash
            portfolio['cash'] -= position_value
            
            # Record trade
            trade = {
                'trade_id': position['id'],
                'entry_date': position['entry_date'],
                'entry_price': current_price,
                'shares': shares,
                'position_type': signal_type,
                'signal_strength': signal_strength,
                'signal_quality': signal_quality,
                'position_size': position_size
            }
            
            portfolio['trades'].append(trade)
            
            logger.info(f"Executed {signal_type} signal: {shares:.2f} shares at ${current_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def _calculate_portfolio_value(self, portfolio: Dict, current_price: float) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            portfolio: Portfolio state
            current_price: Current asset price
            
        Returns:
            Portfolio value
        """
        total_value = portfolio['cash']
        
        for position_id, position in portfolio['positions'].items():
            if position['type'] == 'BUY':
                position_value = position['shares'] * current_price
            else:  # SELL signal (short position)
                position_value = position['shares'] * (2 * position['entry_price'] - current_price)
            
            total_value += position_value
        
        return total_value
    
    def _check_position_exits(self, portfolio: Dict, current_price: float, current_date: datetime, 
                             max_hold_days: int, stop_loss_pct: float, take_profit_pct: float):
        """
        Check for position exits based on stop loss, take profit, and time-based rules.
        
        Args:
            portfolio: Portfolio state
            current_price: Current asset price
            current_date: Current date
            max_hold_days: Maximum days to hold position
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        positions_to_close = []
        
        for position_id, position in portfolio['positions'].items():
            entry_price = position['entry_price']
            entry_date = position['entry_date']
            position_type = position['type']
            
            # Calculate current P&L
            if position_type == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SELL signal (short position)
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss
            if pnl_pct <= -stop_loss_pct:
                positions_to_close.append((position_id, 'stop_loss', pnl_pct))
                continue
            
            # Check take profit
            if pnl_pct >= take_profit_pct:
                positions_to_close.append((position_id, 'take_profit', pnl_pct))
                continue
            
            # Check time-based exit
            naive_current_date = make_naive(current_date)
            naive_entry_date = make_naive(entry_date)
            days_held = (naive_current_date - naive_entry_date).days
            if days_held >= max_hold_days:
                positions_to_close.append((position_id, 'time_exit', pnl_pct))
                continue
        
        # Close positions
        for position_id, exit_reason, pnl_pct in positions_to_close:
            self._close_position(portfolio, position_id, current_price, current_date, exit_reason, pnl_pct)
    
    def _close_position(self, portfolio: Dict, position_id: int, current_price: float, 
                       current_date: datetime, exit_reason: str, pnl_pct: float):
        """
        Close a position and update portfolio.
        
        Args:
            portfolio: Portfolio state
            position_id: Position ID to close
            current_price: Current asset price
            current_date: Current date
            exit_reason: Reason for exit
            pnl_pct: P&L percentage
        """
        if position_id not in portfolio['positions']:
            return
        
        position = portfolio['positions'][position_id]
        shares = position['shares']
        entry_price = position['entry_price']
        
        # Calculate exit value
        exit_value = shares * current_price
        
        # Update cash
        portfolio['cash'] += exit_value
        
        # Calculate actual P&L
        if position['type'] == 'BUY':
            actual_pnl = exit_value - (shares * entry_price)
        else:  # SELL signal (short position)
            actual_pnl = (shares * entry_price) - exit_value
        
        # Update trade record
        for trade in portfolio['trades']:
            if trade['trade_id'] == position_id:
                naive_current_date = make_naive(current_date)
                naive_entry_date = make_naive(trade['entry_date'])
                trade.update({
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'pnl': actual_pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'duration': (naive_current_date - naive_entry_date).days
                })
                break
        
        # Remove position
        del portfolio['positions'][position_id]
        
        logger.info(f"Closed position {position_id}: {exit_reason}, P&L: {actual_pnl:.2f} ({pnl_pct:.2%})")
    
    def _calculate_performance_metrics(
        self,
        daily_values: List[float],
        daily_returns: List[float],
        benchmark_returns: List[float],
        portfolio: Dict,
        config: Dict
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            daily_values: List of daily portfolio values
            daily_returns: List of daily returns
            benchmark_returns: List of benchmark returns
            portfolio: Portfolio state
            config: Trading configuration
            
        Returns:
            Performance metrics dictionary
        """
        if not daily_values:
            return self._empty_results()
        
        # Convert to numpy arrays
        daily_values = np.array(daily_values)
        daily_returns = np.array(daily_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # Calculate basic metrics
        total_return = (daily_values[-1] - daily_values[0]) / daily_values[0]
        
        # Calculate volatility (annualized)
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(daily_values)
        drawdown = (daily_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calculate win rate
        trades = portfolio.get('trades', [])
        if trades:
            # Calculate actual win rate from completed trades
            completed_trades = [t for t in trades if 'pnl_pct' in t]
            if completed_trades:
                winning_trades = [t for t in completed_trades if t['pnl_pct'] > 0]
                win_rate = len(winning_trades) / len(completed_trades)
            else:
                win_rate = 0.0
        else:
            win_rate = 0.0
        
        # Calculate benchmark comparison
        if len(benchmark_returns) > 1:
            benchmark_total_return = np.prod(1 + benchmark_returns) - 1
            excess_return = total_return - benchmark_total_return
        else:
            benchmark_total_return = 0
            excess_return = 0
        
        # Calculate additional risk metrics
        if len(daily_returns) > 1:
            # VaR and CVaR (95%)
            var_95 = np.percentile(daily_returns, 5)
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            
            # Skewness and Kurtosis
            skewness = np.mean(((daily_returns - np.mean(daily_returns)) / np.std(daily_returns)) ** 3) if np.std(daily_returns) > 0 else 0
            kurtosis = np.mean(((daily_returns - np.mean(daily_returns)) / np.std(daily_returns)) ** 4) if np.std(daily_returns) > 0 else 0
            
            # Calmar ratio
            calmar_ratio = (total_return * 252 / len(daily_returns)) / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            var_95 = 0.0
            cvar_95 = 0.0
            skewness = 0.0
            kurtosis = 0.0
            calmar_ratio = 0.0
        
        # Create results with proper structure for walk-forward validation
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'benchmark_return': benchmark_total_return,
            'excess_return': excess_return,
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'calmar_ratio': calmar_ratio,
            'cumulative_returns': pd.Series(daily_values / daily_values[0] - 1),
            'benchmark_returns': pd.Series(np.cumprod(1 + benchmark_returns) - 1),
            'daily_returns': pd.Series(daily_returns),
            'returns': pd.Series(daily_returns),  # For walk-forward compatibility
            'trades': trades,
            'portfolio_values': pd.Series(daily_values)
        }
        
        return results
    
    def _empty_results(self) -> Dict:
        """
        Return empty results structure.
        
        Returns:
            Empty results dictionary
        """
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'benchmark_return': 0.0,
            'excess_return': 0.0,
            'volatility': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'calmar_ratio': 0.0,
            'cumulative_returns': pd.Series(),
            'benchmark_returns': pd.Series(),
            'daily_returns': pd.Series(),
            'returns': pd.Series(),
            'trades': [],
            'portfolio_values': pd.Series()
        } 