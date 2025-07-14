"""
Enhanced Stress Testing with Realistic Historical and Hypothetical Scenarios
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedStressTestManager:
    """
    Enhanced stress testing with realistic scenarios and comprehensive metrics.
    """
    
    def __init__(self):
        """Initialize the enhanced stress test manager."""
        self.scenarios = {
            'historical': [
                {
                    'label': 'COVID-19 Crash',
                    'start': '2020-02-15',
                    'end': '2020-04-30',
                    'description': 'Global pandemic market crash'
                },
                {
                    'label': 'Russia-Ukraine War',
                    'start': '2022-02-20',
                    'end': '2022-04-15',
                    'description': 'Geopolitical conflict impact'
                },
                {
                    'label': '2008 Financial Crisis',
                    'start': '2008-09-01',
                    'end': '2009-03-31',
                    'description': 'Global financial crisis'
                },
                {
                    'label': '2014 Oil Price Collapse',
                    'start': '2014-06-01',
                    'end': '2015-01-31',
                    'description': 'OPEC supply shock'
                }
            ],
            'hypothetical': [
                {
                    'label': 'Oil Price Collapse (-30%)',
                    'shock_pct': -0.30,
                    'duration_days': 30,
                    'description': 'Hypothetical oil price crash'
                },
                {
                    'label': 'VIX Spike (+15 points)',
                    'vix_change': 15.0,
                    'duration_days': 10,
                    'description': 'Market volatility spike'
                },
                {
                    'label': 'Dollar Strength (+10%)',
                    'dxy_change': 0.10,
                    'duration_days': 20,
                    'description': 'USD appreciation scenario'
                },
                {
                    'label': 'Gold Rally (+20%)',
                    'gold_change': 0.20,
                    'duration_days': 15,
                    'description': 'Safe haven demand surge'
                }
            ]
        }
        
    def fetch_stress_data(self, start_date: str, end_date: str) -> Dict:
        """
        Fetch comprehensive data for stress testing.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with stress test data
        """
        logger.info(f"Fetching enhanced stress test data from {start_date} to {end_date}")
        
        symbols = ['XOP', 'XLE', 'USO', 'BNO', 'SPY', '^VIX', '^DXY', 'GC=F']
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=start_date, end=end_date)
                # Ensure index is datetime and timezone-naive
                hist_data.index = pd.to_datetime(hist_data.index).tz_localize(None)
                
                # Calculate additional metrics
                hist_data['returns'] = hist_data['Close'].pct_change()
                hist_data['volatility'] = hist_data['returns'].rolling(20).std()
                hist_data['drawdown'] = self._calculate_drawdown(hist_data['Close'])
                
                data[symbol] = hist_data
                logger.info(f"Successfully fetched data for {symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return data
    
    def _calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling drawdown."""
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown
    
    def _calculate_stress_metrics(self, returns: pd.Series, prices: pd.Series) -> Dict:
        """
        Calculate comprehensive stress test metrics.
        
        Args:
            returns: Return series
            prices: Price series
            
        Returns:
            Dictionary with stress metrics
        """
        if len(returns) < 2:
            return {
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'time_to_recovery': 0,
                'worst_day': 0.0,
                'best_day': 0.0
            }
        
        # Basic metrics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1)
        max_drawdown = self._calculate_drawdown(prices).min()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Time to recovery
        rolling_max = prices.expanding().max()
        underwater = prices < rolling_max
        recovery_periods = underwater.sum() if underwater.any() else 0
        
        # Daily extremes
        worst_day = returns.min()
        best_day = returns.max()
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'time_to_recovery': recovery_periods,
            'worst_day': worst_day,
            'best_day': best_day
        }
    
    def simulate_historical_scenario(self, data: Dict, scenario: Dict) -> Dict:
        """
        Simulate historical stress scenario.
        
        Args:
            data: Market data
            scenario: Scenario definition
            
        Returns:
            Scenario results
        """
        start_date = pd.to_datetime(scenario['start']).tz_localize(None)
        end_date = pd.to_datetime(scenario['end']).tz_localize(None)
        
        results = {}
        for symbol, symbol_data in data.items():
            # Filter data for scenario period
            mask = (symbol_data.index >= start_date) & (symbol_data.index <= end_date)
            scenario_data = symbol_data[mask]
            
            if len(scenario_data) > 0:
                metrics = self._calculate_stress_metrics(
                    scenario_data['returns'], 
                    scenario_data['Close']
                )
                results[symbol] = {
                    'metrics': metrics,
                    'data': scenario_data,
                    'period_days': len(scenario_data)
                }
        
        return results
    
    def simulate_hypothetical_scenario(self, data: Dict, scenario: Dict, 
                                     base_date: str = '2023-01-01') -> Dict:
        """
        Simulate hypothetical stress scenario.
        
        Args:
            data: Market data
            scenario: Scenario definition
            base_date: Base date for simulation
            
        Returns:
            Scenario results
        """
        base_dt = pd.to_datetime(base_date).tz_localize(None)
        duration_days = scenario.get('duration_days', 30)
        
        # Create hypothetical shock
        shock_data = {}
        for symbol, symbol_data in data.items():
            # Get base period data
            base_period = symbol_data[symbol_data.index >= base_dt].head(duration_days)
            
            if len(base_period) == 0:
                continue
            
            # Apply shock based on scenario type
            modified_data = base_period.copy()
            
            if 'shock_pct' in scenario:  # Oil price shock
                if symbol in ['XOP', 'XLE', 'USO', 'BNO']:
                    shock_factor = 1 + scenario['shock_pct']
                    modified_data['Close'] = modified_data['Close'] * shock_factor
                    modified_data['returns'] = modified_data['Close'].pct_change()
            
            elif 'vix_change' in scenario:  # VIX spike
                if symbol == '^VIX':
                    modified_data['Close'] = modified_data['Close'] + scenario['vix_change']
                elif symbol in ['XOP', 'XLE', 'USO', 'BNO', 'SPY']:
                    # Correlate with VIX (negative correlation)
                    vix_impact = -0.3 * scenario['vix_change'] / 100
                    modified_data['Close'] = modified_data['Close'] * (1 + vix_impact)
                    modified_data['returns'] = modified_data['Close'].pct_change()
            
            elif 'dxy_change' in scenario:  # Dollar strength
                if symbol == '^DXY':
                    modified_data['Close'] = modified_data['Close'] * (1 + scenario['dxy_change'])
                elif symbol in ['XOP', 'XLE', 'USO', 'BNO']:
                    # Oil typically weakens with strong dollar
                    dxy_impact = -0.2 * scenario['dxy_change']
                    modified_data['Close'] = modified_data['Close'] * (1 + dxy_impact)
                    modified_data['returns'] = modified_data['Close'].pct_change()
            
            elif 'gold_change' in scenario:  # Gold rally
                if symbol == 'GC=F':
                    modified_data['Close'] = modified_data['Close'] * (1 + scenario['gold_change'])
                elif symbol in ['XOP', 'XLE', 'USO', 'BNO']:
                    # Oil may benefit from safe haven flows
                    gold_impact = 0.1 * scenario['gold_change']
                    modified_data['Close'] = modified_data['Close'] * (1 + gold_impact)
                    modified_data['returns'] = modified_data['Close'].pct_change()
            
            # Recalculate metrics
            modified_data['volatility'] = modified_data['returns'].rolling(20).std()
            modified_data['drawdown'] = self._calculate_drawdown(modified_data['Close'])
            
            metrics = self._calculate_stress_metrics(
                modified_data['returns'], 
                modified_data['Close']
            )
            
            shock_data[symbol] = {
                'metrics': metrics,
                'data': modified_data,
                'period_days': len(modified_data)
            }
        
        return shock_data
    
    def run_rolling_stress_analysis(self, data: Dict, window_days: int = 60) -> Dict:
        """
        Run rolling window stress analysis.
        
        Args:
            data: Market data
            window_days: Rolling window size
            
        Returns:
            Rolling analysis results
        """
        logger.info(f"Running rolling stress analysis with {window_days}-day windows")
        
        rolling_results = {}
        for symbol, symbol_data in data.items():
            if len(symbol_data) < window_days:
                continue
            
            rolling_metrics = []
            for i in range(window_days, len(symbol_data)):
                window_data = symbol_data.iloc[i-window_days:i]
                metrics = self._calculate_stress_metrics(
                    window_data['returns'], 
                    window_data['Close']
                )
                metrics['date'] = symbol_data.index[i]
                rolling_metrics.append(metrics)
            
            if rolling_metrics:
                rolling_results[symbol] = pd.DataFrame(rolling_metrics)
                rolling_results[symbol].set_index('date', inplace=True)
        
        return rolling_results
    
    def create_stress_visualizations(self, scenario_results: Dict, 
                                   save_path: str = None) -> Dict:
        """
        Create comprehensive stress test visualizations.
        
        Args:
            scenario_results: Results from stress scenarios
            save_path: Path to save plots
            
        Returns:
            Dictionary with plot paths
        """
        plots = {}
        
        # Scenario comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stress Test Scenario Comparison', fontsize=16)
        
        metrics_to_plot = ['total_return', 'max_drawdown', 'volatility', 'var_95']
        metric_names = ['Total Return', 'Max Drawdown', 'Volatility', 'VaR (95%)']
        
        for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[i//2, i%2]
            
            # Collect data for plotting
            scenarios = []
            values = []
            symbols = []
            
            for scenario_name, scenario_data in scenario_results.items():
                for symbol, symbol_data in scenario_data.items():
                    if 'metrics' in symbol_data:
                        scenarios.append(scenario_name)
                        values.append(symbol_data['metrics'].get(metric, 0))
                        symbols.append(symbol)
            
            if values:
                # Create grouped bar plot
                df_plot = pd.DataFrame({
                    'Scenario': scenarios,
                    'Value': values,
                    'Symbol': symbols
                })
                
                # Pivot for grouped bars
                pivot_df = df_plot.pivot(index='Symbol', columns='Scenario', values='Value')
                pivot_df.plot(kind='bar', ax=ax, alpha=0.8)
                
                ax.set_title(name)
                ax.set_ylabel(name)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            comparison_path = f"{save_path}_scenario_comparison.png"
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plots['scenario_comparison'] = comparison_path
        plt.close()
        
        # Performance curves for key scenarios
        key_scenarios = ['COVID-19 Crash', 'Oil Price Collapse (-30%)', 'VIX Spike (+15 points)']
        
        for scenario_name in key_scenarios:
            if scenario_name in scenario_results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'{scenario_name} - Performance Analysis', fontsize=16)
                
                scenario_data = scenario_results[scenario_name]
                
                # Price evolution
                ax1 = axes[0, 0]
                for symbol, symbol_data in scenario_data.items():
                    if 'data' in symbol_data:
                        prices = symbol_data['data']['Close']
                        normalized_prices = prices / prices.iloc[0]
                        ax1.plot(normalized_prices.index, normalized_prices.values, 
                               label=symbol, linewidth=2)
                ax1.set_title('Price Evolution (Normalized)')
                ax1.set_ylabel('Normalized Price')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Drawdown
                ax2 = axes[0, 1]
                for symbol, symbol_data in scenario_data.items():
                    if 'data' in symbol_data:
                        drawdown = symbol_data['data']['drawdown']
                        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                                       alpha=0.6, label=symbol)
                ax2.set_title('Drawdown Analysis')
                ax2.set_ylabel('Drawdown')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Volatility
                ax3 = axes[1, 0]
                for symbol, symbol_data in scenario_data.items():
                    if 'data' in symbol_data:
                        volatility = symbol_data['data']['volatility']
                        ax3.plot(volatility.index, volatility.values, 
                               label=symbol, linewidth=2)
                ax3.set_title('Volatility Evolution')
                ax3.set_ylabel('Volatility')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Returns distribution
                ax4 = axes[1, 1]
                for symbol, symbol_data in scenario_data.items():
                    if 'data' in symbol_data:
                        returns = symbol_data['data']['returns'].dropna()
                        ax4.hist(returns, bins=20, alpha=0.6, label=symbol, density=True)
                ax4.set_title('Returns Distribution')
                ax4.set_xlabel('Returns')
                ax4.set_ylabel('Density')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_path:
                    scenario_path = f"{save_path}_{scenario_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}.png"
                    plt.savefig(scenario_path, dpi=300, bbox_inches='tight')
                    plots[f'scenario_{scenario_name}'] = scenario_path
                plt.close()
        
        return plots
    
    def perform_enhanced_stress_testing(self, returns_df: pd.DataFrame, 
                                      events: List[Dict] = None) -> Dict:
        """
        Perform comprehensive enhanced stress testing.
        
        Args:
            returns_df: Strategy returns DataFrame
            events: List of custom events to test
            
        Returns:
            Comprehensive stress test results
        """
        logger.info("Starting enhanced stress testing...")
        
        # Use default events if none provided
        if events is None:
            events = [
                {"label": "COVID-19 Crash", "start": "2020-02-15", "end": "2020-04-30"},
                {"label": "Russia-Ukraine War", "start": "2022-02-20", "end": "2022-04-15"},
                {"label": "Hypothetical Oil Collapse", "shock_pct": -0.30},
                {"label": "VIX Spike Scenario", "vix_change": 15.0},
            ]
        
        # Fetch comprehensive data
        start_date = '2015-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = self.fetch_stress_data(start_date, end_date)
        
        # Run historical scenarios
        historical_results = {}
        for scenario in self.scenarios['historical']:
            try:
                results = self.simulate_historical_scenario(data, scenario)
                historical_results[scenario['label']] = results
                logger.info(f"Completed historical scenario: {scenario['label']}")
            except Exception as e:
                logger.warning(f"Failed historical scenario {scenario['label']}: {e}")
        
        # Run hypothetical scenarios
        hypothetical_results = {}
        for scenario in self.scenarios['hypothetical']:
            try:
                results = self.simulate_hypothetical_scenario(data, scenario)
                hypothetical_results[scenario['label']] = results
                logger.info(f"Completed hypothetical scenario: {scenario['label']}")
            except Exception as e:
                logger.warning(f"Failed hypothetical scenario {scenario['label']}: {e}")
        
        # Run rolling analysis
        rolling_results = self.run_rolling_stress_analysis(data)
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = {**historical_results, **hypothetical_results}
        plots = self.create_stress_visualizations(all_results, f"data/plots/enhanced_stress_test_{timestamp}")
        
        # Compile summary statistics
        summary_stats = self._compile_summary_statistics(all_results)
        
        results = {
            'historical_scenarios': historical_results,
            'hypothetical_scenarios': hypothetical_results,
            'rolling_analysis': rolling_results,
            'summary_statistics': summary_stats,
            'plots': plots,
            'scenarios_tested': len(all_results),
            'data_period': f"{start_date} to {end_date}"
        }
        
        logger.info("Enhanced stress testing completed successfully")
        
        return results
    
    def _compile_summary_statistics(self, scenario_results: Dict) -> Dict:
        """
        Compile summary statistics across all scenarios.
        
        Args:
            scenario_results: Results from all scenarios
            
        Returns:
            Summary statistics
        """
        all_metrics = []
        
        for scenario_name, scenario_data in scenario_results.items():
            for symbol, symbol_data in scenario_data.items():
                if 'metrics' in symbol_data:
                    metrics = symbol_data['metrics'].copy()
                    metrics['scenario'] = scenario_name
                    metrics['symbol'] = symbol
                    all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        df_metrics = pd.DataFrame(all_metrics)
        
        summary = {
            'worst_drawdown': df_metrics['max_drawdown'].min(),
            'average_drawdown': df_metrics['max_drawdown'].mean(),
            'worst_return': df_metrics['total_return'].min(),
            'average_return': df_metrics['total_return'].mean(),
            'highest_volatility': df_metrics['volatility'].max(),
            'average_volatility': df_metrics['volatility'].mean(),
            'worst_var': df_metrics['var_95'].min(),
            'average_var': df_metrics['var_95'].mean(),
            'scenarios_count': len(scenario_results),
            'symbols_tested': df_metrics['symbol'].nunique()
        }
        
        return summary 