"""
Main entry point for Smart Signal Filtering system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import yaml
from pathlib import Path

from data import EquityDataCollector, NewsDataCollector
from models import LLMSentimentAnalyzer
from signals import MultiCriteriaFilter
from trading import ContrarianTrader, WalkForwardValidator
from risk import RiskManager, FactorExposureAnalyzer, StressTestManager
from agents import AIAgent
from models.bias_detection import BiasDetector

logger = logging.getLogger(__name__)


class SmartSignalFilter:
    """
    Main class for Smart Signal Filtering system.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the Smart Signal Filter system.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        
        # Load configurations
        self.trading_config = self._load_config("trading.yaml")
        self.sentiment_config = self._load_config("sentiment.yaml")
        self.data_config = self._load_config("data_sources.yaml")
        
        # Initialize components with enhanced capabilities
        self.equity_collector = EquityDataCollector()
        self.news_collector = NewsDataCollector()
        self.sentiment_analyzer = LLMSentimentAnalyzer(use_llm=False)  # Disable LLM for speed
        self.signal_filter = MultiCriteriaFilter()
        self.trader = ContrarianTrader()
        self.risk_manager = RiskManager()
        self.ai_agent = AIAgent(use_llm=False)  # Disable LLM for speed
        
        # Initialize new components
        self.walk_forward_validator = WalkForwardValidator()
        self.factor_analyzer = FactorExposureAnalyzer()
        self.stress_manager = StressTestManager()
        self.bias_detector = BiasDetector()
        
        # Results storage
        self.results = {}
        
    def _load_config(self, filename: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            filename: Configuration filename
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config {filename}: {e}")
            return {}
    
    def run_pipeline(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        symbols: Optional[List[str]] = None,
        backtest: bool = True,
        save_results: bool = True,
        sentiment_threshold: float = 0.2,
        volatility_threshold: float = 0.03,
        reliability_threshold: float = 0.4,
        news_sources: Optional[List[str]] = None
    ) -> Dict:
        """
        Run the complete Smart Signal Filtering pipeline.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            symbols: List of symbols to analyze (default: XOP and related)
            backtest: Whether to run backtest
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting Smart Signal Filtering pipeline")
        
        # Set default symbols if not provided
        if symbols is None:
            symbols = ["XOP", "XLE", "USO", "BNO", "SPY"]
        
        try:
            # Step 1: Collect equity data
            logger.info("Step 1: Collecting equity data")
            equity_data = self.equity_collector.fetch_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            if not equity_data:
                raise ValueError("No equity data collected")
            
            # Step 2: Collect news data
            logger.info("Step 2: Collecting news data")
            news_data = self.news_collector.collect_news(
                start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(end_date, "%Y-%m-%d"),
                max_articles=500  # Cap at 1000 articles for speed
            )
            
            # Step 3: Analyze sentiment
            logger.info("Step 3: Analyzing sentiment")
            if not news_data.empty:
                news_with_sentiment = self.sentiment_analyzer.analyze_batch(news_data)
            else:
                news_with_sentiment = pd.DataFrame()
            
            # Step 4: Generate signals
            logger.info("Step 4: Generating signals")
            
            # Update config with user parameters
            signal_config = self.trading_config.copy()
            signal_config['sentiment_threshold'] = sentiment_threshold
            signal_config['volatility_threshold'] = volatility_threshold
            signal_config['reliability_threshold'] = reliability_threshold
            signal_config['news_sources'] = news_sources or ["Reuters", "Al Jazeera", "Bloomberg", "BBC"]
            
            signals = self.signal_filter.generate_signals(
                equity_data=equity_data,
                news_data=news_with_sentiment,
                config=signal_config
            )
            
            # Step 5: Run trading strategy
            logger.info("Step 5: Running trading strategy")
            if backtest:
                trading_results = self.trader.backtest_strategy(
                    equity_data=equity_data,
                    signals=signals,
                    config=self.trading_config
                )
            else:
                trading_results = self.trader.live_trading(
                    equity_data=equity_data,
                    signals=signals,
                    config=self.trading_config
                )
            
            # Step 6: Risk analysis
            logger.info("Step 6: Risk analysis")
            risk_analysis = self.risk_manager.analyze_risk(
                trading_results=trading_results,
                equity_data=equity_data,
                config=self.trading_config
            )
            
            # Step 7: Factor exposure analysis
            logger.info("Step 7: Factor exposure analysis")
            factor_results = self.factor_analyzer.run_complete_analysis(start_date, end_date)
            
            # Step 8: Walk-forward validation
            logger.info("Step 8: Walk-forward validation")
            wf_results = self.walk_forward_validator.run_walk_forward_validation(
                data=equity_data,
                strategy_func=self.trader.backtest_strategy,
                equity_data=equity_data,
                signals=signals,
                config=self.trading_config
            )
            
            # Step 9: Stress testing
            logger.info("Step 9: Stress testing")
            stress_results = self.stress_manager.run_all_stress_tests(trading_results)
            
            # Step 10: Bias detection
            logger.info("Step 10: Bias detection")
            if not news_with_sentiment.empty:
                bias_results = self.bias_detector.detect_sentiment_bias(
                    sentiment_scores=news_with_sentiment['sentiment_score'].tolist(),
                    sources=news_with_sentiment['source'].tolist()
                )
            else:
                bias_results = {}
            
            # Step 11: AI agent analysis
            logger.info("Step 11: AI agent analysis")
            agent_insights = self.ai_agent.analyze_market(
                equity_data=equity_data,
                news_data=news_with_sentiment,
                signals=signals,
                trading_results=trading_results
            )
            
            # Step 11.5: AI agent adaptive learning and recommendations
            logger.info("Step 11.5: AI agent adaptive learning")
            
            # Observe current performance metrics
            if trading_results:
                performance_metrics = {
                    'total_return': trading_results.get('total_return', 0),
                    'sharpe_ratio': trading_results.get('sharpe_ratio', 0),
                    'max_drawdown': trading_results.get('max_drawdown', 0),
                    'win_rate': trading_results.get('win_rate', 0),
                    'total_trades': trading_results.get('total_trades', 0),
                    'volatility': trading_results.get('volatility', 0)
                }
                self.ai_agent.observe_metrics(performance_metrics)
            
            # Generate parameter recommendations
            recommendations = self.ai_agent.recommend_adjustments()
            
            # Generate topic reweighting suggestions
            topic_suggestions = self.ai_agent.suggest_topic_reweighting(news_with_sentiment)
            
            # Compile results
            self.results = {
                'equity_data': equity_data,
                'news_data': news_with_sentiment,
                'signals': signals,
                'trading_results': trading_results,
                'risk_analysis': risk_analysis,
                'factor_analysis': factor_results,
                'walk_forward_results': wf_results,
                'stress_test_results': stress_results,
                'bias_analysis': bias_results,
                'agent_insights': agent_insights,
                'agent_recommendations': recommendations,
                'topic_suggestions': topic_suggestions,
                'metadata': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbols': symbols,
                    'backtest': backtest,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Save results if requested
            if save_results:
                self._save_results()
            
            logger.info("Pipeline completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _save_results(self):
        """
        Save results to files.
        """
        try:
            # Create results directory
            results_dir = Path("data/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save equity data
            if self.results.get('equity_data'):
                for symbol, data in self.results['equity_data'].items():
                    filename = f"equity_{symbol}_{timestamp}.csv"
                    data.to_csv(results_dir / filename)
            
            # Save news data
            if not self.results.get('news_data').empty:
                filename = f"news_{timestamp}.csv"
                self.results['news_data'].to_csv(results_dir / filename, index=False)
            
            # Save signals
            if self.results.get('signals') is not None:
                filename = f"signals_{timestamp}.csv"
                self.results['signals'].to_csv(results_dir / filename, index=False)
            
            # Save trading results
            if self.results.get('trading_results'):
                filename = f"trading_results_{timestamp}.json"
                with open(results_dir / filename, 'w') as f:
                    import json
                    json.dump(self.results['trading_results'], f, indent=2, default=str)
            
            # Save summary
            summary = self.get_summary()
            filename = f"summary_{timestamp}.json"
            with open(results_dir / filename, 'w') as f:
                import json
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_summary(self) -> Dict:
        """
        Get summary of results.
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {}
        
        summary = {
            'metadata': self.results.get('metadata', {}),
            'data_summary': {},
            'performance_summary': {},
            'risk_summary': {},
            'agent_summary': {}
        }
        
        # Data summary
        if self.results.get('equity_data'):
            summary['data_summary']['equity_symbols'] = list(self.results['equity_data'].keys())
            summary['data_summary']['equity_records'] = {
                symbol: len(data) for symbol, data in self.results['equity_data'].items()
            }
        
        if not self.results.get('news_data').empty:
            summary['data_summary']['news_articles'] = len(self.results['news_data'])
            summary['data_summary']['news_sources'] = self.results['news_data']['source'].nunique()
        
        # Performance summary
        if self.results.get('trading_results'):
            trading_results = self.results['trading_results']
            summary['performance_summary'] = {
                'total_return': trading_results.get('total_return', 0),
                'sharpe_ratio': trading_results.get('sharpe_ratio', 0),
                'max_drawdown': trading_results.get('max_drawdown', 0),
                'win_rate': trading_results.get('win_rate', 0),
                'total_trades': trading_results.get('total_trades', 0)
            }
        
        # Risk summary
        if self.results.get('risk_analysis'):
            risk_analysis = self.results['risk_analysis']
            summary['risk_summary'] = {
                'var_95': risk_analysis.get('var_95', 0),
                'cvar_95': risk_analysis.get('cvar_95', 0),
                'volatility': risk_analysis.get('volatility', 0),
                'beta': risk_analysis.get('beta', 0)
            }
        
        # Agent summary
        if self.results.get('agent_insights'):
            agent_insights = self.results['agent_insights']
            summary['agent_summary'] = {
                'novelty_score': agent_insights.get('novelty_score', 0),
                'confidence': agent_insights.get('confidence', 0),
                'recommendations': agent_insights.get('recommendations', [])
            }
        
        return summary
    
    def plot_results(self, save_plots: bool = True):
        """
        Generate and display plots of results with proper timestamps and overwriting.
        
        Args:
            save_plots: Whether to save plots to files
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results:
                logger.warning("No results to plot")
                return
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create plots directory
            plots_dir = Path("data/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Smart Signal Filtering Results - {timestamp}', fontsize=16)
            
            # Plot 1: Equity prices with volume
            if self.results.get('equity_data'):
                ax1 = axes[0, 0]
                for symbol in ['XOP', 'XLE', 'USO', 'SPY']:
                    if symbol in self.results['equity_data']:
                        data = self.results['equity_data'][symbol]
                        if 'Close' in data.columns:
                            # Normalize to starting price for comparison
                            normalized_prices = data['Close'] / data['Close'].iloc[0]
                            ax1.plot(data.index, normalized_prices, label=symbol, alpha=0.8, linewidth=2)
                ax1.set_title('Normalized Equity Prices')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Normalized Price')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Sentiment over time with confidence intervals
            if not self.results.get('news_data').empty:
                ax2 = axes[0, 1]
                news_data = self.results['news_data']
                if 'sentiment_score' in news_data.columns:
                    # Group by date and calculate average sentiment with confidence
                    daily_sentiment = news_data.groupby(news_data['published_date'].dt.date).agg({
                        'sentiment_score': ['mean', 'std', 'count']
                    }).reset_index()
                    daily_sentiment.columns = ['date', 'mean_sentiment', 'std_sentiment', 'article_count']
                    
                    # Plot with confidence intervals
                    ax2.plot(daily_sentiment['date'], daily_sentiment['mean_sentiment'], 
                            marker='o', alpha=0.7, linewidth=2, label='Average Sentiment')
                    
                    # Add confidence intervals
                    ax2.fill_between(daily_sentiment['date'], 
                                   daily_sentiment['mean_sentiment'] - daily_sentiment['std_sentiment'],
                                   daily_sentiment['mean_sentiment'] + daily_sentiment['std_sentiment'],
                                   alpha=0.3, label='±1 Std Dev')
                    
                    ax2.set_title('Daily Average Sentiment with Confidence Intervals')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Sentiment Score')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
            
            # Plot 3: Trading performance with drawdown
            if self.results.get('trading_results'):
                ax3 = axes[1, 0]
                trading_results = self.results['trading_results']
                if 'cumulative_returns' in trading_results:
                    # Plot cumulative returns
                    ax3.plot(trading_results['cumulative_returns'].index, 
                            trading_results['cumulative_returns'].values, 
                            label='Strategy', linewidth=2, color='blue')
                    
                    # Add benchmark if available
                    if 'benchmark_returns' in trading_results:
                        ax3.plot(trading_results['benchmark_returns'].index,
                                trading_results['benchmark_returns'].values,
                                label='Benchmark (SPY)', alpha=0.7, linewidth=2, color='orange')
                    
                    # Add drawdown overlay
                    if 'drawdown' in trading_results:
                        ax3_twin = ax3.twinx()
                        ax3_twin.fill_between(trading_results['drawdown'].index,
                                            trading_results['drawdown'].values,
                                            0, alpha=0.3, color='red', label='Drawdown')
                        ax3_twin.set_ylabel('Drawdown (%)', color='red')
                        ax3_twin.tick_params(axis='y', labelcolor='red')
                    
                    ax3.set_title('Cumulative Returns with Drawdown')
                    ax3.set_xlabel('Date')
                    ax3.set_ylabel('Cumulative Return')
                    ax3.legend(loc='upper left')
                    ax3.grid(True, alpha=0.3)
            
            # Plot 4: Signal analysis and risk metrics
            ax4 = axes[1, 1]
            
            # Create subplot for signals and risk
            if self.results.get('signals') is not None and not self.results['signals'].empty:
                signals = self.results['signals']
                
                # Plot signal strength over time
                if 'signal_strength' in signals.columns:
                    ax4.plot(signals.index, signals['signal_strength'], 
                            marker='o', alpha=0.7, label='Signal Strength', color='green')
                
                # Add risk metrics as bars
                if self.results.get('risk_analysis'):
                    risk_analysis = self.results['risk_analysis']
                    risk_metrics = ['Volatility', 'VaR (95%)', 'CVaR (95%)', 'Max Drawdown']
                    risk_values = [
                        risk_analysis.get('volatility', 0) * 100,  # Convert to percentage
                        risk_analysis.get('var_95', 0) * 100,
                        risk_analysis.get('cvar_95', 0) * 100,
                        risk_analysis.get('max_drawdown', 0) * 100
                    ]
                    
                    # Create secondary axis for risk metrics
                    ax4_twin = ax4.twinx()
                    bars = ax4_twin.bar(risk_metrics, risk_values, alpha=0.6, color='red')
                    ax4_twin.set_ylabel('Risk Metrics (%)', color='red')
                    ax4_twin.tick_params(axis='y', labelcolor='red')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, risk_values):
                        ax4_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                    f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
                
                ax4.set_title('Signal Strength and Risk Metrics')
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Signal Strength')
                ax4.legend(loc='upper left')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot with timestamp
            if save_plots:
                plot_filename = f"results_{timestamp}.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {plot_path}")
            
            plt.show()
            
            # Generate additional specialized plots
            self._generate_specialized_plots(timestamp, plots_dir)
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _generate_specialized_plots(self, timestamp: str, plots_dir: Path):
        """
        Generate specialized plots for detailed analysis.
        
        Args:
            timestamp: Timestamp for filenames
            plots_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot 1: Factor exposure analysis
            if self.results.get('factor_analysis'):
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Factor Exposure Analysis - {timestamp}', fontsize=16)
                
                factor_results = self.results['factor_analysis']
                
                # PCA explained variance
                if 'pca_results' in factor_results:
                    pca_results = factor_results['pca_results']
                    if 'explained_variance_ratio' in pca_results:
                        axes[0, 0].plot(range(1, len(pca_results['explained_variance_ratio']) + 1),
                                       pca_results['explained_variance_ratio'], 'bo-')
                        axes[0, 0].set_title('PCA Explained Variance Ratio')
                        axes[0, 0].set_xlabel('Principal Component')
                        axes[0, 0].set_ylabel('Explained Variance Ratio')
                        axes[0, 0].grid(True, alpha=0.3)
                
                # Factor loadings heatmap
                if 'factor_loadings' in factor_results:
                    loadings_df = factor_results['factor_loadings']
                    sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, ax=axes[0, 1])
                    axes[0, 1].set_title('Factor Loadings Heatmap')
                
                # Risk decomposition
                if 'risk_decomposition' in factor_results:
                    risk_decomp = factor_results['risk_decomposition']
                    risk_sources = list(risk_decomp.keys())
                    risk_values = list(risk_decomp.values())
                    
                    axes[1, 0].pie(risk_values, labels=risk_sources, autopct='%1.1f%%')
                    axes[1, 0].set_title('Risk Decomposition')
                
                # Factor returns over time
                if 'factor_returns' in factor_results:
                    factor_returns = factor_results['factor_returns']
                    for factor in factor_returns.columns:
                        axes[1, 1].plot(factor_returns.index, factor_returns[factor], 
                                       label=factor, alpha=0.7)
                    axes[1, 1].set_title('Factor Returns Over Time')
                    axes[1, 1].set_xlabel('Date')
                    axes[1, 1].set_ylabel('Factor Return')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / f"factor_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 2: Walk-forward validation results
            if self.results.get('walk_forward_results'):
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Walk-Forward Validation Results - {timestamp}', fontsize=16)
                
                wf_results = self.results['walk_forward_results']
                
                # Train vs Test performance
                if 'splits' in wf_results:
                    splits = wf_results['splits']
                    train_returns = [split['train_performance'].get('total_return', 0) for split in splits]
                    test_returns = [split['test_performance'].get('total_return', 0) for split in splits]
                    
                    x = range(len(splits))
                    axes[0, 0].plot(x, train_returns, 'bo-', label='Train', alpha=0.7)
                    axes[0, 0].plot(x, test_returns, 'ro-', label='Test', alpha=0.7)
                    axes[0, 0].set_title('Train vs Test Returns by Split')
                    axes[0, 0].set_xlabel('Split')
                    axes[0, 0].set_ylabel('Total Return')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                
                # Sharpe ratio comparison
                if 'splits' in wf_results:
                    train_sharpes = [split['train_performance'].get('sharpe_ratio', 0) for split in splits]
                    test_sharpes = [split['test_performance'].get('sharpe_ratio', 0) for split in splits]
                    
                    axes[0, 1].plot(x, train_sharpes, 'bo-', label='Train', alpha=0.7)
                    axes[0, 1].plot(x, test_sharpes, 'ro-', label='Test', alpha=0.7)
                    axes[0, 1].set_title('Train vs Test Sharpe Ratios')
                    axes[0, 1].set_xlabel('Split')
                    axes[0, 1].set_ylabel('Sharpe Ratio')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Consistency metrics
                if 'consistency_metrics' in wf_results:
                    consistency = wf_results['consistency_metrics']
                    metrics = ['Consistency Ratio', 'Stability Score']
                    values = [consistency.get('consistency_ratio', 0), consistency.get('stability', 0)]
                    
                    bars = axes[1, 0].bar(metrics, values, alpha=0.7, color=['blue', 'green'])
                    axes[1, 0].set_title('Walk-Forward Consistency Metrics')
                    axes[1, 0].set_ylabel('Score')
                    
                    # Add value labels
                    for bar, value in zip(bars, values):
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{value:.2f}', ha='center', va='bottom')
                
                # Performance distribution
                if 'splits' in wf_results:
                    all_test_returns = [split['test_performance'].get('total_return', 0) for split in splits]
                    axes[1, 1].hist(all_test_returns, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[1, 1].set_title('Distribution of Test Returns')
                    axes[1, 1].set_xlabel('Total Return')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / f"walk_forward_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 3: Signal analysis
            if self.results.get('signals') is not None and not self.results['signals'].empty:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Signal Analysis - {timestamp}', fontsize=16)
                
                signals = self.results['signals']
                
                # Signal strength distribution
                if 'signal_strength' in signals.columns:
                    axes[0, 0].hist(signals['signal_strength'], bins=20, alpha=0.7, color='green', edgecolor='black')
                    axes[0, 0].set_title('Signal Strength Distribution')
                    axes[0, 0].set_xlabel('Signal Strength')
                    axes[0, 0].set_ylabel('Frequency')
                    axes[0, 0].grid(True, alpha=0.3)
                
                # Signal quality vs strength
                if 'signal_quality' in signals.columns and 'signal_strength' in signals.columns:
                    axes[0, 1].scatter(signals['signal_strength'], signals['signal_quality'], alpha=0.6)
                    axes[0, 1].set_title('Signal Quality vs Strength')
                    axes[0, 1].set_xlabel('Signal Strength')
                    axes[0, 1].set_ylabel('Signal Quality')
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Signal frequency over time
                if 'signal_type' in signals.columns:
                    signal_counts = signals['signal_type'].value_counts()
                    axes[1, 0].pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%')
                    axes[1, 0].set_title('Signal Type Distribution')
                
                # Signal timing analysis
                if 'signal_type' in signals.columns:
                    buy_signals = signals[signals['signal_type'] == 'BUY']
                    sell_signals = signals[signals['signal_type'] == 'SELL']
                    
                    axes[1, 1].plot(buy_signals.index, buy_signals['signal_strength'], 
                                   'go', label='Buy Signals', alpha=0.7)
                    axes[1, 1].plot(sell_signals.index, sell_signals['signal_strength'], 
                                   'ro', label='Sell Signals', alpha=0.7)
                    axes[1, 1].set_title('Signal Timing Analysis')
                    axes[1, 1].set_xlabel('Date')
                    axes[1, 1].set_ylabel('Signal Strength')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / f"signal_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Specialized plots generated with timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"Error generating specialized plots: {e}")
    
    def generate_report(self, output_path: str = "data/report.html"):
        """
        Generate HTML report of results.
        
        Args:
            output_path: Path to save HTML report
        """
        try:
            summary = self.get_summary()
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Smart Signal Filtering Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Smart Signal Filtering Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Metadata</h2>
                    <p><strong>Date Range:</strong> {summary.get('metadata', {}).get('start_date', 'N/A')} to {summary.get('metadata', {}).get('end_date', 'N/A')}</p>
                    <p><strong>Symbols:</strong> {', '.join(summary.get('metadata', {}).get('symbols', []))}</p>
                    <p><strong>Mode:</strong> {'Backtest' if summary.get('metadata', {}).get('backtest', False) else 'Live Trading'}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metric">
                        <strong>Total Return:</strong> 
                        <span class="{'positive' if summary.get('performance_summary', {}).get('total_return', 0) > 0 else 'negative'}">
                            {summary.get('performance_summary', {}).get('total_return', 0):.2%}
                        </span>
                    </div>
                    <div class="metric">
                        <strong>Sharpe Ratio:</strong> {summary.get('performance_summary', {}).get('sharpe_ratio', 0):.3f}
                    </div>
                    <div class="metric">
                        <strong>Max Drawdown:</strong> 
                        <span class="negative">{summary.get('performance_summary', {}).get('max_drawdown', 0):.2%}</span>
                    </div>
                    <div class="metric">
                        <strong>Win Rate:</strong> {summary.get('performance_summary', {}).get('win_rate', 0):.1%}
                    </div>
                    <div class="metric">
                        <strong>Total Trades:</strong> {summary.get('performance_summary', {}).get('total_trades', 0)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Risk Analysis</h2>
                    <div class="metric">
                        <strong>Volatility:</strong> {summary.get('risk_summary', {}).get('volatility', 0):.3f}
                    </div>
                    <div class="metric">
                        <strong>VaR (95%):</strong> {summary.get('risk_summary', {}).get('var_95', 0):.3f}
                    </div>
                    <div class="metric">
                        <strong>CVaR (95%):</strong> {summary.get('risk_summary', {}).get('cvar_95', 0):.3f}
                    </div>
                    <div class="metric">
                        <strong>Beta:</strong> {summary.get('risk_summary', {}).get('beta', 0):.3f}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Data Summary</h2>
                    <p><strong>News Articles:</strong> {summary.get('data_summary', {}).get('news_articles', 0)}</p>
                    <p><strong>News Sources:</strong> {summary.get('data_summary', {}).get('news_sources', 0)}</p>
                    <p><strong>Equity Records:</strong> {summary.get('data_summary', {}).get('equity_records', {})}</p>
                </div>
                
                <div class="section">
                    <h2>AI Agent Insights</h2>
                    <p><strong>Novelty Score:</strong> {summary.get('agent_summary', {}).get('novelty_score', 0):.3f}</p>
                    <p><strong>Confidence:</strong> {summary.get('agent_summary', {}).get('confidence', 0):.3f}</p>
                    <p><strong>Recommendations:</strong></p>
                    <ul>
                        {''.join([f'<li>{rec}</li>' for rec in summary.get('agent_summary', {}).get('recommendations', [])])}
                    </ul>
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Signal Filtering System')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', default=['XOP', 'XLE', 'USO', 'BNO', 'SPY'], help='Symbols to analyze')
    parser.add_argument('--no-backtest', action='store_true', help='Run live trading instead of backtest')
    parser.add_argument('--save-results', action='store_true', default=True, help='Save results to files')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    
    args = parser.parse_args()
    
    # Initialize system
    filter_system = SmartSignalFilter()
    
    # Run pipeline
    results = filter_system.run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
        backtest=not args.no_backtest,
        save_results=args.save_results
    )
    
    # Generate plots if requested
    if args.plot:
        filter_system.plot_results()
    
    # Generate report if requested
    if args.report:
        filter_system.generate_report()
    
    # Print summary
    summary = filter_system.get_summary()
    print("\n" + "="*50)
    print("SMART SIGNAL FILTERING RESULTS")
    print("="*50)
    print(f"Date Range: {summary.get('metadata', {}).get('start_date', 'N/A')} to {summary.get('metadata', {}).get('end_date', 'N/A')}")
    print(f"Total Return: {summary.get('performance_summary', {}).get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {summary.get('performance_summary', {}).get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {summary.get('performance_summary', {}).get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {summary.get('performance_summary', {}).get('win_rate', 0):.1%}")
    print("="*50)


if __name__ == "__main__":
    main() 