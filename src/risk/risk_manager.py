"""
Risk manager for analyzing risk and performing stress tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages risk analysis and stress testing.
    """
    
    def __init__(self):
        """
        Initialize the risk manager.
        """
        pass
    
    def analyze_risk(
        self,
        trading_results: Dict,
        equity_data: Dict[str, pd.DataFrame],
        config: Dict
    ) -> Dict:
        """
        Analyze risk metrics for the trading strategy.
        
        Args:
            trading_results: Trading results dictionary
            equity_data: Dictionary of equity DataFrames
            config: Trading configuration
            
        Returns:
            Risk analysis dictionary
        """
        if not trading_results or 'daily_returns' not in trading_results:
            return self._empty_risk_analysis()
        
        daily_returns = trading_results['daily_returns']
        
        if daily_returns.empty:
            return self._empty_risk_analysis()
        
        # Calculate basic risk metrics
        risk_metrics = self._calculate_basic_risk_metrics(daily_returns)
        
        # Calculate VaR and CVaR
        var_metrics = self._calculate_var_metrics(daily_returns)
        
        # Calculate factor exposures
        factor_exposures = self._calculate_factor_exposures(trading_results, equity_data)
        
        # Calculate turnover
        turnover = self._calculate_turnover(trading_results)
        
        # Perform stress tests
        stress_tests = self._perform_stress_tests(daily_returns, equity_data)
        
        # Compile results
        risk_analysis = {
            **risk_metrics,
            **var_metrics,
            'factor_exposures': factor_exposures,
            'stress_tests': stress_tests,
            'turnover': turnover
        }
        
        return risk_analysis
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at the given confidence level.
        """
        if returns.empty:
            return 0.0
        var_metrics = self._calculate_var_metrics(returns)
        if confidence_level == 0.95:
            return var_metrics.get('var_95', 0.0)
        elif confidence_level == 0.99:
            return var_metrics.get('var_99', 0.0)
        else:
            # Interpolate or fallback
            return float(np.percentile(returns.dropna(), 100 * (1 - confidence_level)))
    
    def _calculate_basic_risk_metrics(self, daily_returns: pd.Series) -> Dict:
        """
        Calculate basic risk metrics.
        
        Args:
            daily_returns: Series of daily returns
            
        Returns:
            Dictionary of risk metrics
        """
        if daily_returns.empty:
            return {}
        
        returns = daily_returns.dropna()
        
        metrics = {
            'volatility': returns.std() * np.sqrt(252),  # Annualized
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'calmar_ratio': returns.mean() / abs(self._calculate_max_drawdown(returns)) if self._calculate_max_drawdown(returns) != 0 else 0
        }
        
        return metrics
    
    def _calculate_var_metrics(self, daily_returns: pd.Series) -> Dict:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Args:
            daily_returns: Series of daily returns
            
        Returns:
            Dictionary of VaR metrics
        """
        if daily_returns.empty:
            return {}
        
        returns = daily_returns.dropna()
        
        # Historical VaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Parametric VaR (assuming normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        parametric_var_95 = mean_return - 1.645 * std_return
        parametric_var_99 = mean_return - 2.326 * std_return
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'parametric_var_95': parametric_var_95,
            'parametric_var_99': parametric_var_99
        }
    
    def _calculate_factor_exposures(
        self,
        trading_results: Dict,
        equity_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Calculate factor exposures.
        
        Args:
            trading_results: Trading results
            equity_data: Equity data
            
        Returns:
            Dictionary of factor exposures
        """
        if 'daily_returns' not in trading_results or trading_results['daily_returns'].empty:
            return {}
        
        strategy_returns = trading_results['daily_returns']
        
        exposures = {}
        
        # Market exposure (SPY)
        if 'SPY' in equity_data:
            try:
                market_returns = equity_data['SPY']['Returns'].dropna()
                # Ensure both indices are datetime
                strategy_idx = pd.to_datetime(strategy_returns.index)
                market_idx = pd.to_datetime(market_returns.index)
                common_dates = strategy_idx.intersection(market_idx)
                if len(common_dates) > 30:
                    strategy_aligned = strategy_returns.loc[common_dates]
                    market_aligned = market_returns.loc[common_dates]
                    beta = np.cov(strategy_aligned, market_aligned)[0, 1] / np.var(market_aligned)
                    exposures['market_beta'] = beta
            except Exception as e:
                logger.warning(f"Error calculating market exposure: {e}")
        
        # Oil exposure (USO)
        if 'USO' in equity_data:
            try:
                oil_returns = equity_data['USO']['Returns'].dropna()
                # Ensure both indices are datetime
                strategy_idx = pd.to_datetime(strategy_returns.index)
                oil_idx = pd.to_datetime(oil_returns.index)
                common_dates = strategy_idx.intersection(oil_idx)
                if len(common_dates) > 30:
                    strategy_aligned = strategy_returns.loc[common_dates]
                    oil_aligned = oil_returns.loc[common_dates]
                    oil_beta = np.cov(strategy_aligned, oil_aligned)[0, 1] / np.var(oil_aligned)
                    exposures['oil_beta'] = oil_beta
            except Exception as e:
                logger.warning(f"Error calculating oil exposure: {e}")
        
        # Energy sector exposure (XLE)
        if 'XLE' in equity_data:
            try:
                energy_returns = equity_data['XLE']['Returns'].dropna()
                # Ensure both indices are datetime
                strategy_idx = pd.to_datetime(strategy_returns.index)
                energy_idx = pd.to_datetime(energy_returns.index)
                common_dates = strategy_idx.intersection(energy_idx)
                if len(common_dates) > 30:
                    strategy_aligned = strategy_returns.loc[common_dates]
                    energy_aligned = energy_returns.loc[common_dates]
                    energy_beta = np.cov(strategy_aligned, energy_aligned)[0, 1] / np.var(energy_aligned)
                    exposures['energy_beta'] = energy_beta
            except Exception as e:
                logger.warning(f"Error calculating energy exposure: {e}")
        
        return exposures
    
    def _perform_stress_tests(
        self,
        daily_returns: pd.Series,
        equity_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Perform stress tests on the strategy.
        
        Args:
            daily_returns: Strategy daily returns
            equity_data: Equity data
            
        Returns:
            Dictionary of stress test results
        """
        stress_tests = {}
        
        try:
            # COVID-19 stress test (March 2020)
            covid_start = pd.Timestamp('2020-03-01')
            covid_end = pd.Timestamp('2020-04-30')
            # Ensure index is datetime and remove timezone info for comparison
            daily_returns_idx = pd.to_datetime(daily_returns.index).tz_localize(None)
            covid_returns = daily_returns[(daily_returns_idx >= covid_start) & (daily_returns_idx <= covid_end)]
            
            if not covid_returns.empty:
                stress_tests['covid_19'] = {
                    'period_return': covid_returns.sum(),
                    'max_drawdown': self._calculate_max_drawdown(covid_returns),
                    'volatility': covid_returns.std() * np.sqrt(252)
                }
            
            # Oil crash stress test (2014-2016)
            oil_crash_start = pd.Timestamp('2014-06-01')
            oil_crash_end = pd.Timestamp('2016-02-29')
            oil_crash_returns = daily_returns[(daily_returns_idx >= oil_crash_start) & (daily_returns_idx <= oil_crash_end)]
            
            if not oil_crash_returns.empty:
                stress_tests['oil_crash_2014_2016'] = {
                    'period_return': oil_crash_returns.sum(),
                    'max_drawdown': self._calculate_max_drawdown(oil_crash_returns),
                    'volatility': oil_crash_returns.std() * np.sqrt(252)
                }
            
            # Ukraine conflict stress test (2022)
            ukraine_start = pd.Timestamp('2022-02-01')
            ukraine_end = pd.Timestamp('2022-06-30')
            ukraine_returns = daily_returns[(daily_returns_idx >= ukraine_start) & (daily_returns_idx <= ukraine_end)]
            
            if not ukraine_returns.empty:
                stress_tests['ukraine_conflict_2022'] = {
                    'period_return': ukraine_returns.sum(),
                    'max_drawdown': self._calculate_max_drawdown(ukraine_returns),
                    'volatility': ukraine_returns.std() * np.sqrt(252)
                }
        except Exception as e:
            logger.warning(f"Error in stress tests: {e}")
            # Create empty stress test results
            stress_tests = {}
        
        return stress_tests
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown
        """
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def _calculate_turnover(self, trading_results: Dict) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            trading_results: Trading results dictionary
            
        Returns:
            Turnover rate
        """
        if 'trades' not in trading_results:
            return 0.0
        
        trades = trading_results['trades']
        if not trades:
            return 0.0
        
        # Calculate total trading volume
        total_volume = sum(abs(trade.get('value', 0)) for trade in trades)
        
        # Get average portfolio value
        if 'portfolio_values' in trading_results:
            avg_portfolio_value = trading_results['portfolio_values'].mean()
        else:
            avg_portfolio_value = 100000  # Default value
        
        # Calculate turnover rate
        turnover_rate = total_volume / avg_portfolio_value if avg_portfolio_value > 0 else 0.0
        
        return turnover_rate
    
    def _empty_risk_analysis(self) -> Dict:
        """
        Return empty risk analysis structure.
        
        Returns:
            Empty risk analysis dictionary
        """
        return {
            'volatility': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'cvar_95': 0.0,
            'cvar_99': 0.0,
            'factor_exposures': {},
            'stress_tests': {}
        } 