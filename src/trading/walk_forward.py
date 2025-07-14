"""
Walk-Forward Validation Module

Implements rolling validation and regime testing for the Smart Signal Filtering system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardValidator:
    def __init__(self, train_period: int = 252, test_period: int = 63, step_size: int = 21):
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.results = []

    def create_walk_forward_splits(self, data: pd.DataFrame) -> List[Tuple]:
        logger.info("Creating walk-forward splits")
        splits = []
        total_days = len(data)

        if total_days < 50:
            logger.warning(f"Insufficient data: {total_days} days, need at least 50")
            return []

        if total_days < 252:
            train_period = max(63, total_days // 3)
            test_period = max(21, total_days // 6)
            step_size = max(10, total_days // 12)
        else:
            train_period = self.train_period
            test_period = self.test_period
            step_size = self.step_size

        min_required = train_period + test_period
        if total_days < min_required:
            train_period = max(30, total_days // 4)
            test_period = max(15, total_days // 8)
            step_size = max(5, total_days // 16)
            min_required = train_period + test_period

        train_start = 0
        split_count = 0
        max_splits = 10

        while (train_start + train_period + test_period <= total_days and split_count < max_splits):
            train_end = train_start + train_period
            test_start = train_end
            test_end = min(test_start + test_period, total_days)

            if test_end - test_start >= test_period // 2:
                splits.append((train_start, train_end, test_start, test_end))
                split_count += 1
                logger.debug(f"Split {split_count}: Train {train_start}-{train_end}, Test {test_start}-{test_end}")

            train_start += step_size

            if train_start + train_period + test_period > total_days:
                break

        if not splits and total_days >= 50:
            mid_point = total_days // 2
            splits.append((0, mid_point, mid_point, total_days))
            logger.info("Created single split due to insufficient data")

        if len(splits) < 2 and total_days >= 100:
            split_size = total_days // 2
            splits = [
                (0, split_size, split_size, total_days),
                (split_size // 2, split_size + split_size // 2, split_size + split_size // 2, total_days)
            ]
            logger.info("Created 2 equal splits for validation")

        if not splits and total_days >= 60:
            split_size = total_days // 4
            for i in range(3):
                train_start = i * split_size
                train_end = (i + 1) * split_size
                test_start = train_end
                test_end = min(test_start + split_size, total_days)
                if test_end > test_start:
                    splits.append((train_start, train_end, test_start, test_end))
            logger.info("Created minimal splits for validation")

        logger.info(f"Created {len(splits)} walk-forward splits (train: {train_period}, test: {test_period}, step: {step_size})")
        return splits

    def run_walk_forward_validation(self, data: pd.DataFrame, strategy_func, equity_data: Dict[str, pd.DataFrame], signals: pd.DataFrame, config: Dict) -> Dict:
        logger.info("Running walk-forward validation")
        primary_symbol = config.get('assets', {}).get('primary', 'XOP')

        if primary_symbol not in equity_data:
            logger.error(f"Primary asset {primary_symbol} not found in equity data")
            return {'error': f'Primary asset {primary_symbol} not found'}

        primary_data = equity_data[primary_symbol]
        splits = self.create_walk_forward_splits(primary_data)

        if not splits:
            total_days = len(primary_data)
            split_size = total_days // 3
            splits = []
            for i in range(2):
                train_start = i * split_size
                train_end = (i + 1) * split_size
                test_start = train_end
                test_end = min(test_start + split_size, total_days)
                if test_end > test_start:
                    splits.append((train_start, train_end, test_start, test_end))

        if not splits:
            logger.error("Failed to create any walk-forward splits")
            return {'error': 'Failed to create walk-forward splits'}

        results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            logger.info(f"Split {i+1}/{len(splits)}: Train {train_start}-{train_end}, Test {test_start}-{test_end}")
            try:
                train_equity = {
                    symbol: df.iloc[train_start:min(train_end, len(df))]
                    for symbol, df in equity_data.items()
                }
                test_equity = {
                    symbol: df.iloc[test_start:min(test_end, len(df))]
                    for symbol, df in equity_data.items()
                }

                train_signals = signals.iloc[train_start:min(train_end, len(signals))] if not signals.empty else pd.DataFrame()
                test_signals = signals.iloc[test_start:min(test_end, len(signals))] if not signals.empty else pd.DataFrame()

                train_results = strategy_func(equity_data=train_equity, signals=train_signals, config=config)
                test_results = strategy_func(equity_data=test_equity, signals=test_signals, config=config)

                split_result = {
                    'split_id': i + 1,
                    'train_start': primary_data.index[train_start],
                    'train_end': primary_data.index[min(train_end - 1, len(primary_data) - 1)],
                    'test_start': primary_data.index[test_start],
                    'test_end': primary_data.index[min(test_end - 1, len(primary_data) - 1)],
                    'train_performance': self.calculate_performance_metrics(train_results),
                    'test_performance': self.calculate_performance_metrics(test_results),
                }

                results.append(split_result)
            except Exception as e:
                logger.error(f"Error in split {i+1}: {e}")
                results.append({
                    'split_id': i + 1,
                    'error': str(e),
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })

        aggregated_results = self.aggregate_walk_forward_results(results)
        consistency_metrics = self.calculate_consistency_metrics(results)

        return {
            'splits': results,
            'aggregated_results': aggregated_results,
            'consistency_metrics': consistency_metrics,
            'num_splits': len(results),
            'successful_splits': len([r for r in results if 'error' not in r])
        }

    def calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics from strategy results."""
        if not results or 'error' in results:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'calmar_ratio': 0.0
            }
        
        # Extract returns from different possible keys
        returns = None
        if 'returns' in results and not results['returns'].empty:
            returns = results['returns']
        elif 'daily_returns' in results and not results['daily_returns'].empty:
            returns = results['daily_returns']
        elif 'cumulative_returns' in results and not results['cumulative_returns'].empty:
            # Convert cumulative returns to daily returns
            cumulative = results['cumulative_returns']
            returns = cumulative.pct_change().dropna()
        
        if returns is None or returns.empty:
            # Fallback to basic metrics from results
            return {
                'total_return': results.get('total_return', 0.0) * 100,
                'annualized_return': results.get('total_return', 0.0) * 100,
                'volatility': results.get('volatility', 0.0) * 100,
                'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                'max_drawdown': results.get('max_drawdown', 0.0) * 100,
                'win_rate': results.get('win_rate', 0.0) * 100,
                'calmar_ratio': results.get('calmar_ratio', 0.0)
            }
        
        # Calculate metrics from returns series
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        win_rate = (returns > 0).mean()
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        return {
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'calmar_ratio': calmar_ratio,
        }

    def aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        if not results:
            return {'error': 'No results to aggregate'}

        train_metrics = [r['train_performance'] for r in results if 'error' not in r['train_performance']]
        test_metrics = [r['test_performance'] for r in results if 'error' not in r['test_performance']]

        def calculate_stats(metrics_list):
            stats = {}
            for metric in ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'calmar_ratio']:
                values = [m[metric] for m in metrics_list if metric in m]
                if values:
                    stats[f'{metric}_mean'] = np.mean(values)
                    stats[f'{metric}_std'] = np.std(values)
                    stats[f'{metric}_min'] = np.min(values)
                    stats[f'{metric}_max'] = np.max(values)
                    stats[f'{metric}_median'] = np.median(values)
            return stats

        return {
            'train_statistics': calculate_stats(train_metrics),
            'test_statistics': calculate_stats(test_metrics),
        }

    def calculate_consistency_metrics(self, results: List[Dict]) -> Dict:
        test_sharpes = [r['test_performance'].get('sharpe_ratio', 0)
                        for r in results if 'error' not in r['test_performance']]
        if not test_sharpes:
            return {}

        positive_sharpes = sum(1 for s in test_sharpes if s > 0)
        consistency_ratio = positive_sharpes / len(test_sharpes)
        mean_sharpe = np.mean(test_sharpes)
        std_sharpe = np.std(test_sharpes)
        stability = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else np.inf

        return {
            'consistency_ratio': consistency_ratio * 100,
            'stability': stability,
            'mean_test_sharpe': mean_sharpe,
            'std_test_sharpe': std_sharpe
        }

    def _dummy_strategy_func(self, equity_data: Dict[str, pd.DataFrame], signals: pd.DataFrame, config: Dict) -> Dict:
        index = list(equity_data.values())[0].index
        returns = pd.Series(np.random.normal(0.001, 0.02, len(index)), index=index)
        return {'returns': returns}

    def run_regime_analysis(self, data: pd.DataFrame, regime_identifier_func, strategy_func, equity_data: Dict[str, pd.DataFrame], signals: pd.DataFrame, config: Dict) -> Dict:
        logger.info("Running regime analysis")
        regimes = regime_identifier_func(data)

        regime_results = {}
        for regime in regimes.dropna().unique():
            regime_data = data[regimes == regime]
            sub_signals = signals.loc[regime_data.index.intersection(signals.index)]
            result = self.run_walk_forward_validation(
                data=regime_data,
                strategy_func=strategy_func,
                equity_data=equity_data,
                signals=sub_signals,
                config=config
            )
            regime_results[regime] = result

        return regime_results
