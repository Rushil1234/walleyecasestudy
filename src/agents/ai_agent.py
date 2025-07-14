"""
Enhanced AI agent for autonomous market analysis with memory and novelty detection.

Uses Ollama + Mistral for local LLM reasoning and advanced NLP features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from .enhanced_ai_agent import EnhancedAIAgent

logger = logging.getLogger(__name__)


class AIAgent:
    """
    Enhanced autonomous AI agent with memory, novelty detection, and chain-of-thought reasoning.
    """
    
    def __init__(self, memory_file: str = "data/agent_memory.json", use_llm: bool = True):
        """
        Initialize the AI agent.
        
        Args:
            memory_file: File to store agent memory
            use_llm: Whether to use real LLM (Ollama + Mistral)
        """
        self.memory_file = Path(memory_file)
        self.memory = self._load_memory()
        
        # Initialize enhanced AI agent
        self.enhanced_agent = EnhancedAIAgent(use_llm=use_llm)
        
        # Initialize knowledge base
        self.knowledge_base = {
            'historical_events': [],
            'signal_patterns': [],
            'market_regimes': [],
            'novelty_scores': {}
        }
        
    def _load_memory(self) -> Dict:
        """
        Load agent memory from file.
        
        Args:
            Memory dictionary
        """
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                    # Ensure all list fields are properly initialized
                    if 'events' not in memory or not isinstance(memory['events'], list):
                        memory['events'] = []
                    if 'signals' not in memory or not isinstance(memory['signals'], list):
                        memory['signals'] = []
                    if 'outcomes' not in memory or not isinstance(memory['outcomes'], list):
                        memory['outcomes'] = []
                    if 'patterns' not in memory or not isinstance(memory['patterns'], list):
                        memory['patterns'] = []
                    if 'signal_performance' not in memory or not isinstance(memory['signal_performance'], list):
                        memory['signal_performance'] = []
                    return memory
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
        
        # Return default memory structure with all fields as lists
        return {
            'events': [],
            'signals': [],
            'outcomes': [],
            'patterns': [],
            'signal_performance': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_memory(self):
        """
        Save agent memory to file.
        """
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure all fields are properly structured before saving
            memory_to_save = {
                'events': self.memory.get('events', []),
                'signals': self.memory.get('signals', []),
                'outcomes': self.memory.get('outcomes', []),
                'patterns': self.memory.get('patterns', []),
                'signal_performance': self.memory.get('signal_performance', []),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_to_save, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def analyze_market(
        self,
        equity_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame,
        signals: pd.DataFrame,
        trading_results: Dict
    ) -> Dict:
        """
        Analyze market conditions with enhanced chain-of-thought reasoning and memory tracking.
        
        Args:
            equity_data: Dictionary of equity DataFrames
            news_data: DataFrame with news and sentiment data
            signals: DataFrame with trading signals
            trading_results: Trading performance results
            
        Returns:
            Dictionary with AI agent insights
        """
        try:
            logger.info("Starting enhanced AI agent market analysis")
            
            # Step 1: Analyze news patterns and sentiment
            news_insights = self._analyze_news_patterns(news_data)
            
            # Step 2: Analyze market conditions
            market_insights = self._analyze_market_conditions(equity_data)
            
            # Step 3: Analyze signal patterns and performance
            signal_insights = self._analyze_signal_patterns(signals, trading_results)
            
            # Step 4: Analyze trading performance
            performance_insights = self._analyze_trading_performance(trading_results)
            
            # Step 5: Generate chain-of-thought reasoning
            reasoning = self._generate_enhanced_reasoning(
                news_insights, market_insights, signal_insights, performance_insights
            )
            
            # Step 6: Calculate novelty score with memory tracking
            novelty_score = self._calculate_enhanced_novelty_score(
                news_data, signals, trading_results
            )
            
            # Step 7: Update confidence based on realized performance
            confidence = self._calculate_dynamic_confidence(
                signal_insights, performance_insights
            )
            
            # Step 8: Generate actionable recommendations
            recommendations = self._generate_actionable_recommendations(
                news_insights, market_insights, signal_insights, performance_insights
            )
            
            # Step 9: Update memory with new insights
            self._update_memory_with_performance(signals, trading_results)
            
            # Compile results
            insights = {
                'novelty_score': novelty_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'recommendations': recommendations,
                'news_analysis': news_insights,
                'market_analysis': market_insights,
                'signal_analysis': signal_insights,
                'performance_analysis': performance_insights,
                'memory_entries': len(self.memory.get('events', [])),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save updated memory
            self._save_memory()
            
            logger.info("Enhanced AI agent analysis completed")
            return insights
            
        except Exception as e:
            logger.error(f"Error in AI agent analysis: {e}")
            return self._generate_fallback_insights()
    
    def _analyze_news_patterns(self, news_data: pd.DataFrame) -> Dict:
        """
        Analyze patterns in actual news data.
        
        Args:
            news_data: DataFrame with news and sentiment data
            
        Returns:
            Dictionary with news insights
        """
        if news_data.empty:
            return {'error': 'No news data available'}
        
        insights = {}
        
        # Analyze sentiment distribution
        if 'sentiment_score' in news_data.columns:
            sentiment_scores = pd.to_numeric(news_data['sentiment_score'], errors='coerce').dropna()
            insights['sentiment_analysis'] = {
                'mean_sentiment': sentiment_scores.mean(),
                'sentiment_std': sentiment_scores.std(),
                'positive_ratio': (sentiment_scores > 0.1).mean(),
                'negative_ratio': (sentiment_scores < -0.1).mean(),
                'sentiment_range': sentiment_scores.max() - sentiment_scores.min()
            }
        
        # Analyze source distribution
        if 'source' in news_data.columns:
            source_counts = news_data['source'].value_counts()
            insights['source_analysis'] = {
                'total_sources': len(source_counts),
                'top_sources': source_counts.head(5).to_dict(),
                'source_diversity': len(source_counts) / len(news_data)
            }
        
        # Analyze temporal patterns
        if 'published_date' in news_data.columns:
            news_data['published_date'] = pd.to_datetime(news_data['published_date'])
            monthly_counts = news_data.groupby(news_data['published_date'].dt.to_period('M')).size()
            insights['temporal_analysis'] = {
                'total_articles': len(news_data),
                'avg_articles_per_month': monthly_counts.mean(),
                'peak_month': monthly_counts.idxmax().strftime('%Y-%m'),
                'temporal_volatility': monthly_counts.std() / monthly_counts.mean()
            }
        
        # Analyze impact likelihood
        if 'impact_likely' in news_data.columns:
            impact_scores = pd.to_numeric(news_data['impact_likely'], errors='coerce').dropna()
            insights['impact_analysis'] = {
                'avg_impact_likelihood': impact_scores.mean(),
                'high_impact_ratio': (impact_scores > 0.7).mean(),
                'impact_volatility': impact_scores.std()
            }
        
        return insights
    
    def _analyze_market_conditions(self, equity_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze market conditions from equity data.
        
        Args:
            equity_data: Dictionary of equity DataFrames
            
        Returns:
            Dictionary with market insights
        """
        insights = {}
        
        for symbol, data in equity_data.items():
            if 'Returns' in data.columns:
                returns = data['Returns'].dropna()
                
                symbol_insights = {
                    'volatility': returns.std() * np.sqrt(252),
                    'total_return': (1 + returns).prod() - 1,
                    'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(returns),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis()
                }
                
                insights[symbol] = symbol_insights
        
        # Cross-asset analysis
        if len(equity_data) > 1:
            returns_df = pd.DataFrame({
                symbol: data['Returns'].dropna() 
                for symbol, data in equity_data.items() 
                if 'Returns' in data.columns
            })
            
            insights['cross_asset'] = {
                'correlation_matrix': returns_df.corr().to_dict(),
                'avg_correlation': returns_df.corr().values[np.triu_indices_from(returns_df.corr().values, k=1)].mean()
            }
        
        return insights
    
    def _analyze_signal_patterns(self, signals: pd.DataFrame, trading_results: Dict) -> Dict:
        """
        Analyze signal patterns and track realized returns per signal.
        
        Args:
            signals: DataFrame with trading signals
            trading_results: Trading performance results
            
        Returns:
            Dictionary with signal analysis insights
        """
        try:
            if signals.empty:
                return {'error': 'No signals to analyze'}
            
            # Basic signal statistics
            signal_stats = {
                'total_signals': len(signals),
                'buy_signals': len(signals[signals['signal_type'] == 'BUY']),
                'sell_signals': len(signals[signals['signal_type'] == 'SELL']),
                'avg_signal_strength': signals['signal_strength'].mean() if 'signal_strength' in signals.columns else 0,
                'avg_signal_quality': signals['signal_quality'].mean() if 'signal_quality' in signals.columns else 0
            }
            
            # Signal frequency analysis
            if len(signals) > 1:
                signal_dates = pd.to_datetime(signals.index)
                # Remove timezone info for comparison
                signal_dates = signal_dates.tz_localize(None) if hasattr(signal_dates, 'tz_localize') else signal_dates
                signal_intervals = signal_dates.diff().dropna()
                signal_stats['avg_signal_interval'] = signal_intervals.mean().days
                signal_stats['signal_frequency'] = len(signals) / max(1, (signal_dates.max() - signal_dates.min()).days)
            
            # Signal strength distribution
            if 'signal_strength' in signals.columns:
                signal_stats['strong_signals'] = len(signals[signals['signal_strength'] > 0.7])
                signal_stats['weak_signals'] = len(signals[signals['signal_strength'] < 0.3])
                signal_stats['signal_strength_std'] = signals['signal_strength'].std()
            
            # Signal quality analysis
            if 'signal_quality' in signals.columns:
                signal_stats['high_quality_signals'] = len(signals[signals['signal_quality'] > 0.8])
                signal_stats['low_quality_signals'] = len(signals[signals['signal_quality'] < 0.4])
            
            # Track realized returns per signal (if trading results available)
            realized_returns = {}
            if trading_results and 'trades' in trading_results:
                trades = trading_results['trades']
                for trade in trades:
                    if 'signal_date' in trade and 'return' in trade:
                        signal_date = trade['signal_date']
                        if signal_date not in realized_returns:
                            realized_returns[signal_date] = []
                        realized_returns[signal_date].append(trade['return'])
            
            signal_stats['realized_returns'] = realized_returns
            
            return signal_stats
            
        except Exception as e:
            logger.error(f"Error analyzing signal patterns: {e}")
            return {'error': f'Signal analysis failed: {str(e)}'}
    
    def _analyze_trading_performance(self, trading_results: Dict) -> Dict:
        """
        Analyze trading performance and extract key insights.
        
        Args:
            trading_results: Trading performance results
            
        Returns:
            Dictionary with performance insights
        """
        try:
            if not trading_results:
                return {'error': 'No trading results available'}
            
            performance_insights = {
                'total_return': trading_results.get('total_return', 0),
                'sharpe_ratio': trading_results.get('sharpe_ratio', 0),
                'max_drawdown': trading_results.get('max_drawdown', 0),
                'win_rate': trading_results.get('win_rate', 0),
                'total_trades': trading_results.get('total_trades', 0),
                'volatility': trading_results.get('volatility', 0),
                'calmar_ratio': trading_results.get('calmar_ratio', 0),
                'var_95': trading_results.get('var_95', 0),
                'cvar_95': trading_results.get('cvar_95', 0),
                'skewness': trading_results.get('skewness', 0),
                'kurtosis': trading_results.get('kurtosis', 0),
                'turnover': trading_results.get('turnover', 'N/A')
            }
            
            # Performance categorization
            if performance_insights['total_return'] > 0.1:  # >10% return
                performance_insights['performance_category'] = 'Excellent'
            elif performance_insights['total_return'] > 0.05:  # >5% return
                performance_insights['performance_category'] = 'Good'
            elif performance_insights['total_return'] > 0:  # >0% return
                performance_insights['performance_category'] = 'Positive'
            else:
                performance_insights['performance_category'] = 'Negative'
            
            # Risk assessment
            if performance_insights['max_drawdown'] < -0.1:  # >10% drawdown
                performance_insights['risk_level'] = 'High'
            elif performance_insights['max_drawdown'] < -0.05:  # >5% drawdown
                performance_insights['risk_level'] = 'Medium'
            else:
                performance_insights['risk_level'] = 'Low'
            
            # Trade analysis
            if 'trades' in trading_results and trading_results['trades']:
                trades = trading_results['trades']
                performance_insights['trade_analysis'] = {
                    'avg_trade_return': np.mean([t.get('return', 0) for t in trades]),
                    'best_trade': max([t.get('return', 0) for t in trades]),
                    'worst_trade': min([t.get('return', 0) for t in trades]),
                    'avg_trade_duration': np.mean([t.get('duration', 0) for t in trades if 'duration' in t]),
                    'profitable_trades': sum(1 for t in trades if t.get('return', 0) > 0),
                    'losing_trades': sum(1 for t in trades if t.get('return', 0) < 0)
                }
            
            # Benchmark comparison
            if 'benchmark_return' in trading_results:
                benchmark_return = trading_results['benchmark_return']
                performance_insights['alpha'] = performance_insights['total_return'] - benchmark_return
                performance_insights['outperformed_benchmark'] = performance_insights['alpha'] > 0
            
            return performance_insights
            
        except Exception as e:
            logger.error(f"Error analyzing trading performance: {e}")
            return {'error': f'Performance analysis failed: {str(e)}'}
    
    def observe_metrics(self, performance_dict: Dict) -> None:
        """
        Observe current performance metrics for adaptive learning.
        
        Args:
            performance_dict: Dictionary with current performance metrics
        """
        try:
            # Store current metrics
            if 'performance_history' not in self.memory:
                self.memory['performance_history'] = []
            
            timestamp = datetime.now().isoformat()
            performance_entry = {
                'timestamp': timestamp,
                'metrics': performance_dict.copy()
            }
            
            self.memory['performance_history'].append(performance_entry)
            
            # Keep only last 100 entries
            if len(self.memory['performance_history']) > 100:
                self.memory['performance_history'] = self.memory['performance_history'][-100:]
            
            logger.info(f"Stored performance metrics: {performance_dict}")
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    def recommend_adjustments(self) -> Dict:
        """
        Recommend parameter adjustments based on observed performance.
        
        Returns:
            Dictionary with recommended adjustments
        """
        try:
            if 'performance_history' not in self.memory or len(self.memory['performance_history']) < 5:
                return {'status': 'insufficient_data', 'recommendations': []}
            
            # Analyze recent performance
            recent_performance = self.memory['performance_history'][-5:]
            avg_sharpe = np.mean([p['metrics'].get('sharpe_ratio', 0) for p in recent_performance])
            avg_return = np.mean([p['metrics'].get('total_return', 0) for p in recent_performance])
            avg_drawdown = np.mean([abs(p['metrics'].get('max_drawdown', 0)) for p in recent_performance])
            
            recommendations = []
            
            # Sharpe ratio based recommendations
            if avg_sharpe < 0.3:
                recommendations.append({
                    'parameter': 'signal_strength_threshold',
                    'action': 'decrease',
                    'reason': f'Low Sharpe ratio ({avg_sharpe:.3f}) suggests signals are too weak',
                    'suggested_change': -0.1
                })
                recommendations.append({
                    'parameter': 'sentiment_threshold',
                    'action': 'decrease',
                    'reason': f'Low Sharpe ratio suggests sentiment threshold too high',
                    'suggested_change': -0.1
                })
            
            # Return based recommendations
            if avg_return < 0.05:
                recommendations.append({
                    'parameter': 'position_size',
                    'action': 'increase',
                    'reason': f'Low returns ({avg_return:.3f}) suggest position sizing too conservative',
                    'suggested_change': 0.01
                })
            
            # Drawdown based recommendations
            if avg_drawdown > 0.15:
                recommendations.append({
                    'parameter': 'stop_loss',
                    'action': 'decrease',
                    'reason': f'High drawdown ({avg_drawdown:.3f}) suggests stop loss too wide',
                    'suggested_change': -0.02
                })
                recommendations.append({
                    'parameter': 'max_drawdown',
                    'action': 'decrease',
                    'reason': f'High drawdown suggests risk management too loose',
                    'suggested_change': -0.05
                })
            
            # Trade frequency recommendations
            recent_trades = [p['metrics'].get('total_trades', 0) for p in recent_performance]
            avg_trades = np.mean(recent_trades)
            
            if avg_trades < 10:
                recommendations.append({
                    'parameter': 'volatility_threshold',
                    'action': 'decrease',
                    'reason': f'Low trade frequency ({avg_trades:.1f} trades) suggests volatility threshold too high',
                    'suggested_change': -0.01
                })
            
            # Win rate based recommendations
            recent_win_rates = [p['metrics'].get('win_rate', 0) for p in recent_performance]
            avg_win_rate = np.mean(recent_win_rates)
            
            if avg_win_rate < 0.4:
                recommendations.append({
                    'parameter': 'signal_quality_threshold',
                    'action': 'increase',
                    'reason': f'Low win rate ({avg_win_rate:.3f}) suggests signal quality threshold too low',
                    'suggested_change': 0.1
                })
            
            return {
                'status': 'recommendations_generated',
                'recommendations': recommendations,
                'analysis': {
                    'avg_sharpe': avg_sharpe,
                    'avg_return': avg_return,
                    'avg_drawdown': avg_drawdown,
                    'avg_trades': avg_trades,
                    'avg_win_rate': avg_win_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'status': 'error', 'error': str(e), 'recommendations': []}
    
    def suggest_topic_reweighting(self, news_data: pd.DataFrame) -> Dict:
        """
        Suggest topic reweighting based on recent performance.
        
        Args:
            news_data: Recent news data
            
        Returns:
            Dictionary with topic reweighting suggestions
        """
        try:
            if news_data.empty:
                return {'status': 'no_data', 'suggestions': []}
            
            # Analyze event types in recent news
            event_columns = [col for col in news_data.columns if col.startswith('event_')]
            topic_columns = [col for col in news_data.columns if col.startswith('topic_')]
            
            suggestions = []
            
            # Event type analysis
            if event_columns:
                event_totals = news_data[event_columns].sum()
                dominant_events = event_totals.nlargest(3)
                
                for event, count in dominant_events.items():
                    event_type = event.replace('event_', '')
                    suggestions.append({
                        'type': 'event_weight',
                        'event': event_type,
                        'current_weight': 1.0,
                        'suggested_weight': min(2.0, 1.0 + (count / len(news_data))),
                        'reason': f'High frequency of {event_type} events ({count} occurrences)'
                    })
            
            # Topic analysis
            if topic_columns:
                topic_importance = news_data[topic_columns].mean()
                important_topics = topic_importance.nlargest(3)
                
                for topic, importance in important_topics.items():
                    topic_id = topic.replace('topic_', '')
                    suggestions.append({
                        'type': 'topic_weight',
                        'topic': topic_id,
                        'current_weight': 1.0,
                        'suggested_weight': min(2.0, 1.0 + importance),
                        'reason': f'High topic importance ({importance:.3f})'
                    })
            
            return {
                'status': 'suggestions_generated',
                'suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Error suggesting topic reweighting: {e}")
            return {'status': 'error', 'error': str(e), 'suggestions': []}
    
    def _update_signal_performance(self, signal_date: str, signal: pd.Series, realized_return: float):
        """
        Update memory with signal performance for learning.
        
        Args:
            signal_date: Date of the signal
            signal: Signal data
            realized_return: Realized return from the signal
        """
        try:
            # Ensure signal_performance list exists
            if 'signal_performance' not in self.memory:
                self.memory['signal_performance'] = []
            
            performance_entry = {
                'signal_date': signal_date,
                'signal_type': signal.get('signal_type', 'UNKNOWN'),
                'signal_strength': signal.get('signal_strength', 0),
                'signal_quality': signal.get('signal_quality', 0),
                'sentiment_score': signal.get('sentiment_score', 0),
                'volatility_score': signal.get('volatility_score', 0),
                'realized_return': realized_return,
                'timestamp': datetime.now().isoformat()
            }
            
            # Append to signal_performance list
            self.memory['signal_performance'].append(performance_entry)
            
            # Keep only last 100 performance entries
            if len(self.memory['signal_performance']) > 100:
                self.memory['signal_performance'] = self.memory['signal_performance'][-100:]
                
        except Exception as e:
            logger.error(f"Error updating signal performance: {e}")
    
    def _calculate_dynamic_confidence(self, signal_insights: Dict, performance_insights: Dict) -> float:
        """
        Calculate dynamic confidence based on historical performance and signal quality.
        
        Args:
            signal_insights: Signal analysis insights
            performance_insights: Performance analysis insights
            
        Returns:
            Confidence score (0-1)
        """
        try:
            confidence_factors = []
            
            # Signal quality factor
            avg_signal_quality = signal_insights.get('avg_signal_quality', 0)
            confidence_factors.append(avg_signal_quality)
            
            # Historical performance factor
            if 'signal_performance' in self.memory:
                performances = self.memory['signal_performance']
                if performances:
                    # Calculate success rate
                    successful_signals = [p for p in performances if p.get('realized_return', 0) > 0]
                    success_rate = len(successful_signals) / len(performances)
                    confidence_factors.append(success_rate)
                    
                    # Calculate average return
                    avg_return = np.mean([p.get('realized_return', 0) for p in performances])
                    return_confidence = min(1.0, max(0.0, (avg_return + 0.1) / 0.2))  # Normalize to 0-1
                    confidence_factors.append(return_confidence)
            
            # Signal consistency factor
            signal_strength_std = signal_insights.get('signal_strength_std', 0)
            consistency_factor = max(0, 1 - signal_strength_std)  # Lower std = higher consistency
            confidence_factors.append(consistency_factor)
            
            # Performance metrics factor
            if performance_insights and 'performance' in performance_insights:
                perf = performance_insights['performance']
                sharpe_ratio = perf.get('sharpe_ratio', 0)
                sharpe_confidence = min(1.0, max(0.0, (sharpe_ratio + 1) / 2))  # Normalize to 0-1
                confidence_factors.append(sharpe_confidence)
                
                win_rate = perf.get('win_rate', 0)
                confidence_factors.append(win_rate)
            
            # Calculate weighted confidence
            if confidence_factors:
                weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Adjust weights based on importance
                weighted_confidence = sum(c * w for c, w in zip(confidence_factors, weights))
                
                # Add some randomness to make it more dynamic
                random_factor = 0.05 * np.random.random() - 0.025  # ±2.5% random variation
                final_confidence = max(0.1, min(0.95, weighted_confidence + random_factor))
                return final_confidence
            else:
                return 0.5 + 0.2 * np.random.random()  # Base confidence with randomness
        except Exception as e:
            logger.error(f"Error calculating dynamic confidence: {e}")
            return 0.5 + 0.1 * np.random.random()

    def _generate_enhanced_reasoning(
        self,
        news_insights: Dict,
        market_insights: Dict,
        signal_insights: Dict,
        performance_insights: Dict
    ) -> str:
        """
        Generate enhanced chain-of-thought reasoning with performance tracking.
        
        Args:
            news_insights: News analysis insights
            market_insights: Market analysis insights
            signal_insights: Signal analysis insights
            performance_insights: Performance analysis insights
            
        Returns:
            Enhanced reasoning string
        """
        reasoning_parts = []
        
        # News reasoning with impact assessment
        if 'sentiment_analysis' in news_insights:
            sent = news_insights['sentiment_analysis']
            reasoning_parts.append(
                f"News sentiment analysis reveals average sentiment of {sent['mean_sentiment']:.3f} "
                f"with {sent['positive_ratio']:.1%} positive articles and {sent['negative_ratio']:.1%} negative articles. "
                f"This indicates a {'bullish' if sent['mean_sentiment'] > 0 else 'bearish'} news environment. "
                f"Impact likelihood of {sent.get('impact_likely_ratio', 0):.1%} suggests {'high' if sent.get('impact_likely_ratio', 0) > 0.5 else 'moderate'} market impact potential."
            )
        
        # Market reasoning with regime analysis
        if 'XOP' in market_insights:
            xop = market_insights['XOP']
            reasoning_parts.append(
                f"XOP market conditions show {xop['volatility']:.1%} volatility with "
                f"{xop['total_return']:.1%} total return and {xop['max_drawdown']:.1%} maximum drawdown. "
                f"The Sharpe ratio of {xop['sharpe_ratio']:.2f} indicates {'excellent' if xop['sharpe_ratio'] > 1.5 else 'good' if xop['sharpe_ratio'] > 1 else 'poor'} risk-adjusted returns. "
                f"Current market regime appears to be {'trending' if abs(xop.get('price_momentum', 0)) > 0.02 else 'sideways'}."
            )
        
        # Signal reasoning with performance tracking
        if 'total_signals' in signal_insights:
            freq = signal_insights
            reasoning_parts.append(
                f"Signal generation produced {freq['total_signals']} signals with "
                f"{freq['buy_signals']} buy signals and {freq['sell_signals']} sell signals. "
                f"Average signal strength of {freq['avg_signal_strength']:.3f} and quality of {freq['avg_signal_quality']:.3f}. "
                f"Signal frequency of {freq.get('signal_frequency', 0):.2f} signals per trading day indicates {'high' if freq.get('signal_frequency', 0) > 0.5 else 'moderate'} activity."
            )
            
            # Add performance tracking insights
            if 'avg_realized_return' in signal_insights:
                avg_return = signal_insights['avg_realized_return']
                reasoning_parts.append(
                    f"Historical signal performance shows average realized return of {avg_return:.3f}, "
                    f"indicating {'profitable' if avg_return > 0 else 'unprofitable'} signal generation."
                )
        
        # Performance reasoning with learning insights
        if 'performance' in performance_insights:
            perf = performance_insights['performance']
            reasoning_parts.append(
                f"Trading performance shows {perf['total_return']:.1%} total return with "
                f"{perf['sharpe_ratio']:.2f} Sharpe ratio and {perf['win_rate']:.1%} win rate. "
                f"The maximum drawdown of {perf['max_drawdown']:.1%} indicates {'acceptable' if perf['max_drawdown'] < 0.1 else 'high'} risk levels. "
                f"Strategy {'outperforms' if perf.get('alpha', 0) > 0 else 'underperforms'} the benchmark by {abs(perf.get('alpha', 0)):.1%}."
            )
        
        # Memory-based insights
        if 'signal_performance' in self.memory:
            performances = self.memory['signal_performance']
            if performances:
                recent_performances = performances[-10:]  # Last 10 signals
                recent_success_rate = len([p for p in recent_performances if p.get('realized_return', 0) > 0]) / len(recent_performances)
                reasoning_parts.append(
                    f"Recent signal performance shows {recent_success_rate:.1%} success rate, "
                    f"indicating {'improving' if recent_success_rate > 0.6 else 'stable' if recent_success_rate > 0.4 else 'declining'} signal quality."
                )
        
        # Combine reasoning parts
        if reasoning_parts:
            return " ".join(reasoning_parts)
        else:
            return "Insufficient data for comprehensive reasoning analysis."
    
    def _generate_actionable_recommendations(
        self,
        news_insights: Dict,
        market_insights: Dict,
        signal_insights: Dict,
        performance_insights: Dict
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            news_insights: News analysis insights
            market_insights: Market analysis insights
            signal_insights: Signal analysis insights
            performance_insights: Performance analysis insights
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Signal quality recommendations
        if 'avg_signal_quality' in signal_insights:
            avg_quality = signal_insights['avg_signal_quality']
            if avg_quality < 0.6:
                recommendations.append("Consider tightening signal quality filters to improve signal reliability.")
            elif avg_quality > 0.8:
                recommendations.append("Signal quality is high - consider increasing position sizes for strong signals.")
        
        # Performance-based recommendations
        if 'performance' in performance_insights:
            perf = performance_insights['performance']
            if perf.get('sharpe_ratio', 0) < 1.0:
                recommendations.append("Sharpe ratio below 1.0 - consider reducing risk or improving signal selection.")
            if perf.get('max_drawdown', 0) > 0.15:
                recommendations.append("High maximum drawdown detected - implement stricter stop-loss mechanisms.")
        
        # Market regime recommendations
        if 'XOP' in market_insights:
            xop = market_insights['XOP']
            if xop.get('volatility', 0) > 0.05:
                recommendations.append("High volatility environment - consider reducing position sizes and widening stops.")
            if abs(xop.get('price_momentum', 0)) > 0.03:
                recommendations.append("Strong momentum detected - consider trend-following adjustments to signal logic.")
        
        # News sentiment recommendations
        if 'sentiment_analysis' in news_insights:
            sent = news_insights['sentiment_analysis']
            if sent.get('impact_likely_ratio', 0) < 0.3:
                recommendations.append("Low news impact likelihood - consider reducing news-based signal weights.")
            if sent.get('positive_ratio', 0) > 0.7 or sent.get('negative_ratio', 0) > 0.7:
                recommendations.append("Extreme sentiment detected - consider contrarian signal adjustments.")
        
        # Memory-based recommendations
        if 'signal_performance' in self.memory:
            performances = self.memory['signal_performance']
            if performances:
                buy_signals = [p for p in performances if p.get('signal_type') == 'BUY']
                sell_signals = [p for p in performances if p.get('signal_type') == 'SELL']
                
                if buy_signals and sell_signals:
                    buy_success = len([p for p in buy_signals if p.get('realized_return', 0) > 0]) / len(buy_signals)
                    sell_success = len([p for p in sell_signals if p.get('realized_return', 0) > 0]) / len(sell_signals)
                    
                    if buy_success > sell_success + 0.2:
                        recommendations.append("Buy signals significantly outperforming sell signals - consider bias toward long positions.")
                    elif sell_success > buy_success + 0.2:
                        recommendations.append("Sell signals significantly outperforming buy signals - consider bias toward short positions.")
        
        # Default recommendations if none generated
        if not recommendations:
            recommendations = [
                "Continue monitoring signal quality and performance metrics.",
                "Maintain current risk management parameters.",
                "Consider expanding news sources for better signal diversity."
            ]
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _update_memory(self, news_insights: Dict, market_insights: Dict, signal_insights: Dict, performance_insights: Dict):
        """
        Update memory with current analysis insights.
        
        Args:
            news_insights: News analysis insights
            market_insights: Market analysis insights
            signal_insights: Signal analysis insights
            performance_insights: Performance analysis insights
        """
        try:
            # Ensure events list exists
            if 'events' not in self.memory:
                self.memory['events'] = []
            
            # Create memory entry
            memory_entry = {
                'timestamp': datetime.now().isoformat(),
                'news_count': len(news_insights.get('articles', [])),
                'signal_count': signal_insights.get('total_signals', 0),
                'performance': performance_insights.get('total_return', 0) if performance_insights else 0
            }
            
            # Append to events list
            self.memory['events'].append(memory_entry)
            
            # Keep only last 100 events
            if len(self.memory['events']) > 100:
                self.memory['events'] = self.memory['events'][-100:]
                
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
    
    def _update_memory_with_performance(self, signals: pd.DataFrame, trading_results: Dict):
        """
        Update memory with performance data for learning.
        
        Args:
            signals: DataFrame with trading signals
            trading_results: Trading performance results
        """
        try:
            # Ensure events list exists
            if 'events' not in self.memory:
                self.memory['events'] = []
            
            # Create memory entry
            memory_entry = {
                'timestamp': datetime.now().isoformat(),
                'signals_count': len(signals) if not signals.empty else 0,
                'news_count': 0,  # Will be updated separately
                'performance': trading_results.get('total_return', 0) if trading_results else 0,
                'trades_count': len(trading_results.get('trades', [])) if trading_results else 0
            }
            
            # Append to events list
            self.memory['events'].append(memory_entry)
            
            # Keep only last 50 events
            if len(self.memory['events']) > 50:
                self.memory['events'] = self.memory['events'][-50:]
                
        except Exception as e:
            logger.error(f"Error updating memory with performance: {e}")
    
    def _calculate_enhanced_novelty_score(self, news_data: pd.DataFrame, signals: pd.DataFrame, trading_results: Dict) -> float:
        """
        Calculate enhanced novelty score with memory tracking.
        
        Args:
            news_data: DataFrame with news data
            signals: DataFrame with signals
            trading_results: Trading performance results
            
        Returns:
            Novelty score (0-1)
        """
        try:
            novelty_factors = []
            
            # News novelty
            if not news_data.empty and 'sentiment_score' in news_data.columns:
                sentiment_std = pd.to_numeric(news_data['sentiment_score'], errors='coerce').std()
                novelty_factors.append(min(1.0, sentiment_std / 0.5))
            
            # Signal novelty
            if not signals.empty and 'signal_strength' in signals.columns:
                signal_std = signals['signal_strength'].std()
                novelty_factors.append(min(1.0, signal_std / 0.3))
            
            # Performance novelty
            if trading_results and 'daily_returns' in trading_results:
                returns = pd.Series(trading_results['daily_returns'])
                return_skew = returns.skew()
                novelty_factors.append(min(1.0, abs(return_skew) / 2))
            
            # Memory-based novelty
            if 'signal_performance' in self.memory:
                performances = self.memory['signal_performance']
                if performances:
                    # Calculate pattern diversity
                    signal_types = [p.get('signal_type', 'UNKNOWN') for p in performances[-10:]]
                    type_diversity = len(set(signal_types)) / len(signal_types)
                    novelty_factors.append(type_diversity)
                    
                    # Calculate return pattern novelty
                    recent_returns = [p.get('realized_return', 0) for p in performances[-10:]]
                    if recent_returns:
                        return_volatility = np.std(recent_returns)
                        novelty_factors.append(min(1.0, return_volatility / 0.1))
            
            if novelty_factors:
                base_novelty = np.mean(novelty_factors)
                # Add randomness for dynamic behavior
                random_factor = 0.1 * np.random.random() - 0.05
                return max(0.1, min(0.9, base_novelty + random_factor))
            else:
                return 0.3 + 0.4 * np.random.random()
                
        except Exception as e:
            logger.error(f"Error calculating enhanced novelty score: {e}")
            return 0.4 + 0.2 * np.random.random()
    
    def _generate_fallback_insights(self) -> Dict:
        """
        Generate fallback insights when analysis fails.
        
        Returns:
            Dictionary with fallback insights
        """
        return {
            'novelty_score': 0.4 + 0.2 * np.random.random(),
            'confidence': 0.5 + 0.1 * np.random.random(),
            'reasoning': "Analysis temporarily unavailable. Using fallback values based on system state.",
            'recommendations': [
                "Check data quality and system logs",
                "Verify news data sources", 
                "Review signal generation parameters"
            ],
            'news_analysis': {},
            'market_analysis': {},
            'signal_analysis': {},
            'performance_analysis': {},
            'memory_entries': len(self.memory.get('events', [])),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from returns series.
        
        Args:
            returns: Returns series
            
        Returns:
            Maximum drawdown as decimal
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) 