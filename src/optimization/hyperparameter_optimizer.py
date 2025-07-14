"""
Hyperparameter Optimization Module

Uses Optuna for Bayesian optimization to tune signal generation parameters
and maximize Sharpe ratio or walk-forward returns.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
import yaml
from pathlib import Path

# Import our modules
from src.main import SmartSignalFilter
from src.trading.walk_forward import WalkForwardValidator
from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna.
    """
    
    def __init__(self, optimization_target: str = 'sharpe_ratio'):
        """
        Initialize optimizer.
        
        Args:
            optimization_target: Target metric ('sharpe_ratio', 'total_return', 'walk_forward_return')
        """
        self.optimization_target = optimization_target
        self.best_params = None
        self.best_score = -np.inf
        self.trial_history = []
        
    def create_objective_function(self, 
                                start_date: str, 
                                end_date: str, 
                                symbols: List[str],
                                n_trials: int = 50) -> Callable:
        """
        Create objective function for Optuna optimization.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            symbols: List of symbols to analyze
            n_trials: Number of optimization trials
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            try:
                # Define hyperparameter search space
                params = self._suggest_hyperparameters(trial)
                
                # Run pipeline with suggested parameters
                results = self._run_pipeline_with_params(
                    start_date, end_date, symbols, params
                )
                
                if not results:
                    return -np.inf
                
                # Calculate objective score
                score = self._calculate_objective_score(results)
                
                # Store trial results
                trial_info = {
                    'trial_number': trial.number,
                    'params': params,
                    'score': score,
                    'timestamp': datetime.now()
                }
                self.trial_history.append(trial_info)
                
                # Update best if better
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    logger.info(f"New best score: {score:.4f} with params: {params}")
                
                return score
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return -np.inf
        
        return objective
    
    def _suggest_hyperparameters(self, trial) -> Dict:
        """
        Suggest hyperparameters for the trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {
            # Signal thresholds
            'sentiment_threshold': trial.suggest_float('sentiment_threshold', -0.5, 0.5, step=0.1),
            'volatility_threshold': trial.suggest_float('volatility_threshold', 0.005, 0.05, step=0.005),
            'reliability_threshold': trial.suggest_float('reliability_threshold', 0.1, 0.9, step=0.1),
            'signal_strength_threshold': trial.suggest_float('signal_strength_threshold', 0.1, 0.8, step=0.1),
            
            # Signal weights
            'sentiment_weight': trial.suggest_float('sentiment_weight', 0.2, 0.6, step=0.1),
            'volatility_weight': trial.suggest_float('volatility_weight', 0.1, 0.5, step=0.1),
            'reliability_weight': trial.suggest_float('reliability_weight', 0.1, 0.5, step=0.1),
            'impact_weight': trial.suggest_float('impact_weight', 0.05, 0.3, step=0.05),
            
            # Trading parameters
            'position_size': trial.suggest_float('position_size', 0.01, 0.05, step=0.005),
            'stop_loss': trial.suggest_float('stop_loss', 0.03, 0.12, step=0.01),
            'take_profit': trial.suggest_float('take_profit', 0.05, 0.20, step=0.01),
            'holding_period': trial.suggest_int('holding_period', 1, 10),
            
            # Risk management
            'max_drawdown': trial.suggest_float('max_drawdown', 0.10, 0.30, step=0.05),
            'max_positions': trial.suggest_int('max_positions', 3, 8),
            
            # Feature engineering parameters
            'sentiment_window': trial.suggest_int('sentiment_window', 3, 10),
            'volatility_window': trial.suggest_int('volatility_window', 5, 20),
            'correlation_window': trial.suggest_int('correlation_window', 20, 60),
            
            # Walk-forward parameters
            'train_window': trial.suggest_int('train_window', 60, 120),
            'test_window': trial.suggest_int('test_window', 20, 60),
            'step_size': trial.suggest_int('step_size', 10, 30)
        }
        
        return params
    
    def _run_pipeline_with_params(self, 
                                 start_date: str, 
                                 end_date: str, 
                                 symbols: List[str],
                                 params: Dict) -> Optional[Dict]:
        """
        Run the pipeline with given parameters.
        
        Args:
            start_date: Start date
            end_date: End date
            symbols: List of symbols
            params: Hyperparameters
            
        Returns:
            Pipeline results or None if failed
        """
        try:
            # Create configuration with optimized parameters
            config = self._create_config_from_params(params)
            
            # Initialize and run pipeline
            filter_system = SmartSignalFilter()
            results = filter_system.run_pipeline(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                config=config
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed with params {params}: {e}")
            return None
    
    def _create_config_from_params(self, params: Dict) -> Dict:
        """
        Create configuration dictionary from hyperparameters.
        
        Args:
            params: Hyperparameters
            
        Returns:
            Configuration dictionary
        """
        config = {
            'signal_generation': {
                'sentiment_threshold': params['sentiment_threshold'],
                'volatility_threshold': params['volatility_threshold'],
                'reliability_threshold': params['reliability_threshold'],
                'signal_strength_threshold': params['signal_strength_threshold'],
                'weights': {
                    'sentiment': params['sentiment_weight'],
                    'volatility': params['volatility_weight'],
                    'reliability': params['reliability_weight'],
                    'impact': params['impact_weight']
                },
                'windows': {
                    'sentiment': params['sentiment_window'],
                    'volatility': params['volatility_window'],
                    'correlation': params['correlation_window']
                }
            },
            'trading': {
                'position_size': params['position_size'],
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit'],
                'holding_period': params['holding_period'],
                'max_drawdown': params['max_drawdown'],
                'max_positions': params['max_positions']
            },
            'walk_forward': {
                'train_window': params['train_window'],
                'test_window': params['test_window'],
                'step_size': params['step_size']
            }
        }
        
        return config
    
    def _calculate_objective_score(self, results: Dict) -> float:
        """
        Calculate objective score based on optimization target.
        
        Args:
            results: Pipeline results
            
        Returns:
            Objective score
        """
        try:
            if self.optimization_target == 'sharpe_ratio':
                trading_results = results.get('trading_results', {})
                sharpe = trading_results.get('sharpe_ratio', 0)
                return sharpe if not np.isnan(sharpe) else -np.inf
                
            elif self.optimization_target == 'total_return':
                trading_results = results.get('trading_results', {})
                total_return = trading_results.get('total_return', 0)
                return total_return if not np.isnan(total_return) else -np.inf
                
            elif self.optimization_target == 'walk_forward_return':
                wf_results = results.get('walk_forward_results', {})
                if 'splits' in wf_results and wf_results['splits']:
                    test_returns = [split['test_performance'].get('total_return', 0) 
                                  for split in wf_results['splits']]
                    avg_return = np.mean(test_returns)
                    return avg_return if not np.isnan(avg_return) else -np.inf
                return -np.inf
                
            elif self.optimization_target == 'calmar_ratio':
                trading_results = results.get('trading_results', {})
                calmar = trading_results.get('calmar_ratio', 0)
                return calmar if not np.isnan(calmar) else -np.inf
                
            else:
                # Default to Sharpe ratio
                trading_results = results.get('trading_results', {})
                sharpe = trading_results.get('sharpe_ratio', 0)
                return sharpe if not np.isnan(sharpe) else -np.inf
                
        except Exception as e:
            logger.error(f"Error calculating objective score: {e}")
            return -np.inf
    
    def optimize(self, 
                start_date: str, 
                end_date: str, 
                symbols: List[str],
                n_trials: int = 50,
                timeout: int = 3600) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            symbols: List of symbols to analyze
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {self.optimization_target}")
        logger.info(f"Target: {self.optimization_target}, Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Create objective function
        objective = self.create_objective_function(start_date, end_date, symbols, n_trials)
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Compile results
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study,
            'trial_history': self.trial_history,
            'optimization_target': self.optimization_target,
            'n_trials': n_trials,
            'timeout': timeout
        }
        
        logger.info(f"Optimization complete. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return results
    
    def save_results(self, results: Dict, filepath: str = None):
        """
        Save optimization results to file.
        
        Args:
            results: Optimization results
            filepath: Path to save results
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/optimization_results_{timestamp}.yaml"
        
        # Convert to serializable format
        save_data = {
            'best_params': results['best_params'],
            'best_score': float(results['best_score']),
            'optimization_target': results['optimization_target'],
            'n_trials': results['n_trials'],
            'timeout': results['timeout'],
            'timestamp': datetime.now().isoformat(),
            'trial_summary': [
                {
                    'trial_number': t['trial_number'],
                    'score': float(t['score']),
                    'timestamp': t['timestamp'].isoformat()
                }
                for t in results['trial_history']
            ]
        }
        
        # Save to file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(save_data, f, default_flow_style=False)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def plot_optimization_history(self, results: Dict, save_path: str = None):
        """
        Plot optimization history.
        
        Args:
            results: Optimization results
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            study = results['study']
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Hyperparameter Optimization Results - {self.optimization_target}', fontsize=16)
            
            # Plot 1: Optimization history
            axes[0, 0].plot(study.trials_dataframe()['value'])
            axes[0, 0].set_title('Optimization History')
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel(f'{self.optimization_target.replace("_", " ").title()}')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Parameter importance
            if len(study.trials) > 10:
                importance = optuna.importance.get_param_importances(study)
                if importance:
                    params = list(importance.keys())
                    values = list(importance.values())
                    axes[0, 1].barh(params, values)
                    axes[0, 1].set_title('Parameter Importance')
                    axes[0, 1].set_xlabel('Importance')
            
            # Plot 3: Score distribution
            scores = [trial.value for trial in study.trials if trial.value is not None]
            if scores:
                axes[1, 0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Score Distribution')
                axes[1, 0].set_xlabel(f'{self.optimization_target.replace("_", " ").title()}')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Best parameters
            if self.best_params:
                param_names = list(self.best_params.keys())
                param_values = list(self.best_params.values())
                axes[1, 1].barh(param_names, param_values)
                axes[1, 1].set_title('Best Parameters')
                axes[1, 1].set_xlabel('Value')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")


def run_optimization_example():
    """
    Example usage of hyperparameter optimization.
    """
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(optimization_target='sharpe_ratio')
    
    # Define parameters
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    symbols = ["XOP", "XLE", "USO", "BNO", "SPY"]
    
    # Run optimization
    results = optimizer.optimize(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        n_trials=20,  # Reduced for example
        timeout=1800  # 30 minutes
    )
    
    # Save results
    optimizer.save_results(results)
    
    # Plot results
    optimizer.plot_optimization_history(results, "data/optimization_plot.png")
    
    return results


if __name__ == "__main__":
    run_optimization_example() 