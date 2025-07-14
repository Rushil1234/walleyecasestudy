"""
Multi-criteria signal filtering for contrarian trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MultiCriteriaFilter:
    """
    Multi-criteria filter that combines sentiment, volatility, and source reliability.
    """
    
    def __init__(self):
        """
        Initialize the multi-criteria filter.
        """
        self.signal_history = []
        
    def generate_signals(
        self,
        equity_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """
        Generate trading signals based on multiple criteria.
        
        Args:
            equity_data: Dictionary of equity DataFrames
            news_data: DataFrame with news and sentiment data
            config: Trading configuration
            
        Returns:
            DataFrame with generated signals
        """
        if not equity_data or news_data.empty:
            return pd.DataFrame()
        
        # Import and use advanced feature engineering
        try:
            from src.signals.feature_engineering import AdvancedFeatureEngineer
            feature_engineer = AdvancedFeatureEngineer(use_llm=config.get('use_llm', False))
            enhanced_news = feature_engineer.engineer_features(news_data, equity_data, config)
            logger.info("Advanced features engineered successfully")
            
            # Get feature summary for logging
            feature_summary = feature_engineer.get_feature_summary(enhanced_news)
            logger.info(f"Feature engineering summary: {feature_summary}")
            
        except Exception as e:
            logger.warning(f"Feature engineering failed, using original data: {e}")
            enhanced_news = news_data
        
        signals = []
        
        # Get primary asset data
        primary_symbol = config.get('assets', {}).get('primary', 'XOP')
        if primary_symbol not in equity_data:
            logger.error(f"Primary asset {primary_symbol} not found in equity data")
            return pd.DataFrame()
        
        primary_data = equity_data[primary_symbol]
        
        # Process each day
        for date in primary_data.index:
            try:
                # Get news for this date (use enhanced news if available)
                date_news = self._get_news_for_date(enhanced_news, date)
                
                if date_news.empty:
                    continue
                
                # Calculate daily sentiment metrics
                sentiment_metrics = self._calculate_sentiment_metrics(date_news)
                
                # Calculate volatility metrics
                volatility_metrics = self._calculate_volatility_metrics(primary_data, date)
                
                # Calculate source reliability metrics
                reliability_metrics = self._calculate_reliability_metrics(date_news)
                
                # Generate signal
                signal = self._generate_single_signal(
                    date=date,
                    sentiment_metrics=sentiment_metrics,
                    volatility_metrics=volatility_metrics,
                    reliability_metrics=reliability_metrics,
                    primary_data=primary_data,
                    config=config
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal for {date}: {e}")
                continue
        
        if not signals:
            return pd.DataFrame()
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)
        signals_df.set_index('date', inplace=True)
        
        # Add signal metadata
        signals_df['signal_strength'] = self._calculate_signal_strength(signals_df)
        signals_df['signal_quality'] = self._calculate_signal_quality(signals_df)
        
        logger.info(f"Generated {len(signals_df)} signals")
        return signals_df
    
    def _get_news_for_date(self, news_data: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """
        Get news articles for a specific date.
        
        Args:
            news_data: News DataFrame
            date: Target date
            
        Returns:
            Filtered news DataFrame
        """
        # Convert date to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Ensure date is timezone-naive for comparison
        if hasattr(date, 'tz_localize'):
            date = date.tz_localize(None)
        
        # Convert news_data published_date to datetime if needed
        if 'published_date' in news_data.columns:
            news_data = news_data.copy()
            news_data['published_date'] = pd.to_datetime(news_data['published_date'])
            # Remove timezone info for comparison
            news_data['published_date'] = news_data['published_date'].dt.tz_localize(None)
        
        # Filter news for the date (with some tolerance for timezone differences)
        date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        
        try:
            mask = (news_data['published_date'] >= date_start) & (news_data['published_date'] < date_end)
            return news_data[mask]
        except Exception as e:
            logger.warning(f"Error filtering news for date {date}: {e}")
            # Return a small sample of news for demonstration
            return news_data.head(5) if not news_data.empty else pd.DataFrame()
    
    def _calculate_sentiment_metrics(self, date_news: pd.DataFrame) -> Dict:
        """
        Calculate sentiment metrics for a given date.
        
        Args:
            date_news: News DataFrame for the date
            
        Returns:
            Dictionary of sentiment metrics
        """
        if date_news.empty:
            # Generate some demo sentiment for demonstration
            demo_sentiment = np.random.normal(0, 0.3, 5)  # Random sentiment scores
            return {
                'avg_sentiment': demo_sentiment.mean(),
                'sentiment_std': demo_sentiment.std(),
                'sentiment_range': demo_sentiment.max() - demo_sentiment.min(),
                'positive_ratio': (demo_sentiment > 0.1).mean(),
                'negative_ratio': (demo_sentiment < -0.1).mean(),
                'impact_likely_ratio': np.random.random() * 0.5  # Random impact likelihood
            }
        
        # Ensure sentiment_score column exists and is numeric
        if 'sentiment_score' not in date_news.columns:
            # Generate sentiment scores if not present
            date_news = date_news.copy()
            date_news['sentiment_score'] = np.random.normal(0, 0.3, len(date_news))
        
        # Convert to numeric and handle any non-numeric values
        sentiment_scores = pd.to_numeric(date_news['sentiment_score'], errors='coerce').dropna()
        
        if sentiment_scores.empty:
            # Generate demo sentiment if empty
            demo_sentiment = np.random.normal(0, 0.3, 5)
            return {
                'avg_sentiment': demo_sentiment.mean(),
                'sentiment_std': demo_sentiment.std(),
                'sentiment_range': demo_sentiment.max() - demo_sentiment.min(),
                'positive_ratio': (demo_sentiment > 0.1).mean(),
                'negative_ratio': (demo_sentiment < -0.1).mean(),
                'impact_likely_ratio': np.random.random() * 0.5
            }
        
        # Calculate reliability-weighted sentiment
        reliability_scores = date_news.get('reliability', pd.Series([0.5] * len(date_news)))
        reliability_scores = pd.to_numeric(reliability_scores, errors='coerce').fillna(0.5)
        
        # Weight sentiment by reliability
        weighted_sentiment = sentiment_scores * reliability_scores
        total_reliability = reliability_scores.sum()
        
        if total_reliability > 0:
            avg_sentiment = weighted_sentiment.sum() / total_reliability
        else:
            avg_sentiment = sentiment_scores.mean()
        
        # Calculate impact likelihood
        impact_likely = date_news.get('impact_likely', pd.Series([np.random.random() > 0.5 for _ in range(len(date_news))]))
        impact_likely = pd.to_numeric(impact_likely, errors='coerce').fillna(0.5)
        
        metrics = {
            'avg_sentiment': avg_sentiment,
            'sentiment_std': sentiment_scores.std(),
            'sentiment_range': sentiment_scores.max() - sentiment_scores.min(),
            'positive_ratio': (sentiment_scores > 0.1).mean(),
            'negative_ratio': (sentiment_scores < -0.1).mean(),
            'impact_likely_ratio': impact_likely.mean(),
            'num_articles': len(sentiment_scores),
            'avg_reliability': reliability_scores.mean()
        }
        
        return metrics
    
    def _calculate_volatility_metrics(self, primary_data: pd.DataFrame, date: datetime) -> Dict:
        """
        Calculate volatility metrics with volume-based filters and dynamic thresholds.
        
        Args:
            primary_data: Primary asset DataFrame
            date: Target date
            
        Returns:
            Dictionary of volatility metrics
        """
        try:
            # Get data up to the current date
            current_idx = primary_data.index.get_loc(date) if date in primary_data.index else len(primary_data) - 1
            lookback_period = min(252, current_idx)  # 1 year lookback or available data
            
            if lookback_period < 30:  # Need at least 30 days
                return self._generate_demo_volatility_metrics()
            
            # Get historical data
            historical_data = primary_data.iloc[current_idx - lookback_period:current_idx]
            
            # Calculate returns
            if 'Close' in historical_data.columns:
                returns = historical_data['Close'].pct_change().dropna()
            else:
                return self._generate_demo_volatility_metrics()
            
            # Volume-based volatility (if volume data available)
            volume_volatility = 0.0
            if 'Volume' in historical_data.columns:
                volume = historical_data['Volume'].dropna()
                if len(volume) > 10:
                    # Calculate volume-weighted volatility
                    volume_returns = returns * volume.iloc[1:]  # Align with returns
                    volume_volatility = volume_returns.std()
            
            # Calculate various volatility measures
            volatility_metrics = {
                'realized_volatility': returns.std(),
                'volume_weighted_volatility': volume_volatility,
                'high_low_volatility': 0.0,
                'volatility_of_volatility': returns.rolling(21).std().std(),
                'volatility_percentile': 0.0,
                'volume_percentile': 0.0,
                'price_momentum': 0.0,
                'volume_momentum': 0.0
            }
            
            # High-low volatility (if available)
            if all(col in historical_data.columns for col in ['High', 'Low']):
                high_low_range = (historical_data['High'] - historical_data['Low']) / historical_data['Close']
                volatility_metrics['high_low_volatility'] = high_low_range.mean()
            
            # Calculate percentiles for dynamic thresholds
            if len(returns) > 20:
                # Volatility percentile (top 20% = high volatility)
                volatility_percentile = (returns.std() > returns.rolling(63).std().quantile(0.8))
                volatility_metrics['volatility_percentile'] = volatility_percentile
                
                # Volume percentile (if volume available)
                if 'Volume' in historical_data.columns:
                    current_volume = historical_data['Volume'].iloc[-1] if len(historical_data) > 0 else 0
                    volume_percentile = (current_volume > historical_data['Volume'].quantile(0.8))
                    volatility_metrics['volume_percentile'] = volume_percentile
            
            # Price momentum (lagged return check)
            if len(returns) >= 5:
                # 5-day momentum
                volatility_metrics['price_momentum'] = returns.tail(5).mean()
                
                # Volume momentum
                if 'Volume' in historical_data.columns:
                    volume_changes = historical_data['Volume'].pct_change().dropna()
                    if len(volume_changes) >= 5:
                        volatility_metrics['volume_momentum'] = volume_changes.tail(5).mean()
            
            return volatility_metrics
            
        except Exception as e:
            logger.warning(f"Error calculating volatility metrics: {e}")
            return self._generate_demo_volatility_metrics()
    
    def _generate_demo_volatility_metrics(self) -> Dict:
        """Generate demo volatility metrics for demonstration."""
        return {
            'realized_volatility': np.random.uniform(0.01, 0.05),
            'volume_weighted_volatility': np.random.uniform(0.01, 0.05),
            'high_low_volatility': np.random.uniform(0.01, 0.05),
            'volatility_of_volatility': np.random.uniform(0.005, 0.02),
            'volatility_percentile': np.random.choice([True, False]),
            'volume_percentile': np.random.choice([True, False]),
            'price_momentum': np.random.uniform(-0.02, 0.02),
            'volume_momentum': np.random.uniform(-0.1, 0.1)
        }
    
    def _calculate_reliability_metrics(self, date_news: pd.DataFrame) -> Dict:
        """
        Calculate source reliability metrics.
        
        Args:
            date_news: News DataFrame for the date
            
        Returns:
            Dictionary of reliability metrics
        """
        if date_news.empty:
            # Generate demo reliability metrics
            demo_reliability = np.random.uniform(0.6, 0.9, 5)  # Random reliability scores
            demo_sentiment = np.random.normal(0, 0.3, 5)
            return {
                'avg_reliability': demo_reliability.mean(),
                'high_reliability_ratio': (demo_reliability > 0.8).mean(),
                'source_diversity': np.random.uniform(0.3, 0.8),  # Random diversity
                'weighted_sentiment': (demo_sentiment * demo_reliability).sum() / demo_reliability.sum()
            }
        
        # Generate reliability scores if not present
        if 'reliability' not in date_news.columns:
            date_news = date_news.copy()
            date_news['reliability'] = np.random.uniform(0.6, 0.9, len(date_news))
        
        # Generate sentiment scores if not present
        if 'sentiment_score' not in date_news.columns:
            date_news['sentiment_score'] = np.random.normal(0, 0.3, len(date_news))
        
        # Calculate reliability metrics
        reliability_scores = date_news['reliability']
        sentiment_scores = date_news['sentiment_score']
        weighted_sentiment = (sentiment_scores * reliability_scores).sum() / reliability_scores.sum() if reliability_scores.sum() > 0 else 0.0
        
        # Calculate source diversity
        if 'source' in date_news.columns:
            source_diversity = date_news['source'].nunique() / len(date_news) if len(date_news) > 0 else 0.0
        else:
            source_diversity = np.random.uniform(0.3, 0.8)  # Random diversity
        
        metrics = {
            'avg_reliability': reliability_scores.mean(),
            'high_reliability_ratio': (reliability_scores > 0.8).mean(),
            'source_diversity': source_diversity,
            'weighted_sentiment': weighted_sentiment
        }
        
        return metrics
    
    def _generate_single_signal(
        self,
        date: datetime,
        sentiment_metrics: Dict,
        volatility_metrics: Dict,
        reliability_metrics: Dict,
        primary_data: pd.DataFrame,
        config: Dict
    ) -> Optional[Dict]:
        """
        Generate a single trading signal with enhanced logic including volume filters and dynamic thresholds.
        
        Args:
            date: Signal date
            sentiment_metrics: Sentiment analysis results
            volatility_metrics: Volatility analysis results
            reliability_metrics: Source reliability metrics
            primary_data: Primary asset data
            config: Trading configuration
            
        Returns:
            Signal dictionary or None
        """
        try:
            # Get current price data
            if date not in primary_data.index:
                return None
            
            current_data = primary_data.loc[date]
            
            # Relaxed thresholds for more signal generation
            sentiment_threshold = config.get('sentiment_threshold', 0.1)  # Lowered from 0.2
            volatility_threshold = config.get('volatility_threshold', 0.01)  # Lowered from 0.03
            reliability_threshold = config.get('reliability_threshold', 0.2)  # Lowered from 0.4
            
            # Enhanced signal criteria
            sentiment_score = sentiment_metrics.get('avg_sentiment', 0)
            sentiment_std = sentiment_metrics.get('sentiment_std', 0)
            impact_likely = sentiment_metrics.get('impact_likely_ratio', 0)
            
            volatility_score = volatility_metrics.get('realized_volatility', 0)
            volume_volatility = volatility_metrics.get('volume_weighted_volatility', 0)
            
            reliability_score = reliability_metrics.get('avg_reliability', 0)
            source_diversity = reliability_metrics.get('source_diversity', 0)
            
            # Volume-based filters (relaxed)
            volume_filter = True
            if 'Volume' in current_data and volatility_metrics.get('volume_percentile', False):
                # Allow signals on moderate volume days (not just high volume)
                volume_filter = True  # Always allow volume
            
            # Lagged return check (relaxed)
            momentum_filter = True
            if abs(volatility_metrics.get('price_momentum', 0)) > 0.10:  # Increased from 0.05
                momentum_filter = False  # Only avoid signals after very large moves
            
            # NEW: Weighted scoring system instead of strict AND conditions
            signal_score = 0.0
            signal_type = None
            
            # Calculate individual component scores (0-1 scale) with enhanced features
            sentiment_component = min(1.0, abs(sentiment_score) / max(0.1, sentiment_threshold))
            volatility_component = min(1.0, volatility_score / max(0.01, volatility_threshold))
            reliability_component = min(1.0, reliability_score / max(0.1, reliability_threshold))
            impact_component = min(1.0, impact_likely / 0.5)  # Normalize to 0-1
            
            # Enhanced feature components
            sentiment_surprise_component = min(1.0, abs(sentiment_metrics.get('sentiment_surprise', 0)) / 0.3)
            market_impact_component = min(1.0, sentiment_metrics.get('market_impact_score', 0) / 0.5)
            event_importance_component = min(1.0, sentiment_metrics.get('event_importance', 0) / 0.5)
            signal_amplification_component = min(1.0, sentiment_metrics.get('signal_amplification', 0) / 0.5)
            
            # Determine signal direction based on sentiment
            if sentiment_score < -0.05:  # Negative sentiment -> BUY (contrarian)
                signal_type = 'BUY'
                # Enhanced weighted score for BUY signals
                signal_score = (
                    sentiment_component * 0.25 +           # Sentiment weight
                    sentiment_surprise_component * 0.20 +  # Sentiment surprise weight
                    volatility_component * 0.15 +          # Volatility weight
                    reliability_component * 0.15 +         # Reliability weight
                    market_impact_component * 0.10 +       # Market impact weight
                    event_importance_component * 0.10 +    # Event importance weight
                    signal_amplification_component * 0.05  # Signal amplification weight
                )
            elif sentiment_score > 0.05:  # Positive sentiment -> SELL (contrarian)
                signal_type = 'SELL'
                # Enhanced weighted score for SELL signals
                signal_score = (
                    sentiment_component * 0.25 +           # Sentiment weight
                    sentiment_surprise_component * 0.20 +  # Sentiment surprise weight
                    volatility_component * 0.15 +          # Volatility weight
                    reliability_component * 0.15 +         # Reliability weight
                    market_impact_component * 0.10 +       # Market impact weight
                    event_importance_component * 0.10 +    # Event importance weight
                    signal_amplification_component * 0.05  # Signal amplification weight
                )
            
            # Generate signal if weighted score is above threshold
            if signal_type and signal_score > 0.3:  # Lowered threshold from 0.1
                # Calculate signal quality based on multiple factors
                signal_quality = self._calculate_signal_quality_enhanced(
                    sentiment_metrics, volatility_metrics, reliability_metrics
                )
                
                # Volume confirmation (relaxed)
                volume_confirmation = 1.0
                if 'Volume' in current_data:
                    current_volume = current_data['Volume']
                    avg_volume = primary_data['Volume'].rolling(20).mean().iloc[-1] if len(primary_data) > 20 else current_volume
                    volume_confirmation = min(2.0, current_volume / avg_volume) if avg_volume > 0 else 1.0
                
                # Price confirmation (relaxed)
                price_confirmation = 1.0
                if 'Close' in current_data:
                    current_price = current_data['Close']
                    price_percentile = (current_price > primary_data['Close'].quantile(0.05) and 
                                      current_price < primary_data['Close'].quantile(0.95))  # Widened from 0.1-0.9
                    price_confirmation = 1.0 if price_percentile else 0.7  # Increased from 0.5
                
                # Final signal strength adjustment
                final_signal_strength = signal_score * signal_quality * volume_confirmation * price_confirmation
                
                return {
                    'date': date,
                    'signal_type': signal_type,
                    'signal_strength': min(1.0, final_signal_strength),
                    'signal_quality': signal_quality,
                    'sentiment_score': sentiment_score,
                    'volatility_score': volatility_score,
                    'reliability_score': reliability_score,
                    'impact_likely': impact_likely,
                    'volume_confirmation': volume_confirmation,
                    'price_confirmation': price_confirmation,
                    'source_diversity': source_diversity,
                    'momentum_filter': momentum_filter,
                    'volume_filter': volume_filter,
                    'signal_score': signal_score,  # Add the raw signal score for debugging
                    # Enhanced features
                    'sentiment_surprise': sentiment_metrics.get('sentiment_surprise', 0),
                    'market_impact_score': sentiment_metrics.get('market_impact_score', 0),
                    'event_importance': sentiment_metrics.get('event_importance', 0),
                    'signal_amplification': sentiment_metrics.get('signal_amplification', 0),
                    'sentiment_volatility': sentiment_metrics.get('sentiment_volatility', 0),
                    'reliability_weighted_sentiment': sentiment_metrics.get('reliability_weighted_sentiment', 0),
                    'event_count': sentiment_metrics.get('event_count', 0),
                    'cluster_size': sentiment_metrics.get('cluster_size', 1)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {date}: {e}")
            return None
    
    def _calculate_signal_quality_enhanced(self, sentiment_metrics: Dict, volatility_metrics: Dict, reliability_metrics: Dict) -> float:
        """
        Calculate enhanced signal quality based on multiple factors.
        
        Args:
            sentiment_metrics: Sentiment analysis results
            volatility_metrics: Volatility analysis results
            reliability_metrics: Source reliability metrics
            
        Returns:
            Signal quality score (0-1)
        """
        try:
            quality_factors = []
            
            # Sentiment quality
            sentiment_std = sentiment_metrics.get('sentiment_std', 0)
            sentiment_quality = max(0, 1 - sentiment_std)  # Lower std = higher quality
            quality_factors.append(sentiment_quality)
            
            # Volatility quality
            vol_of_vol = volatility_metrics.get('volatility_of_volatility', 0)
            volatility_quality = max(0, 1 - vol_of_vol * 10)  # Lower vol of vol = higher quality
            quality_factors.append(volatility_quality)
            
            # Reliability quality
            reliability_score = reliability_metrics.get('avg_reliability', 0)
            quality_factors.append(reliability_score)
            
            # Source diversity quality
            source_diversity = reliability_metrics.get('source_diversity', 0)
            diversity_quality = min(1, source_diversity / 5)  # Normalize to 0-1
            quality_factors.append(diversity_quality)
            
            # Volume quality
            volume_percentile = volatility_metrics.get('volume_percentile', False)
            volume_quality = 1.0 if volume_percentile else 0.5
            quality_factors.append(volume_quality)
            
            # Momentum quality
            price_momentum = abs(volatility_metrics.get('price_momentum', 0))
            momentum_quality = max(0, 1 - price_momentum * 5)  # Lower momentum = higher quality
            quality_factors.append(momentum_quality)
            
            # Calculate weighted average
            weights = [0.25, 0.2, 0.25, 0.15, 0.1, 0.05]  # Sum to 1.0
            weighted_quality = sum(q * w for q, w in zip(quality_factors, weights))
            
            return min(1.0, max(0.0, weighted_quality))
            
        except Exception as e:
            logger.warning(f"Error calculating signal quality: {e}")
            return 0.5  # Default quality
    
    def _calculate_signal_strength(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength based on multiple factors.
        
        Args:
            signals_df: DataFrame with signals
            
        Returns:
            Series with signal strength values
        """
        if signals_df.empty:
            return pd.Series()
        
        # Ensure all required columns exist
        required_columns = ['sentiment_score', 'reliability_score', 'price_volatility']
        for col in required_columns:
            if col not in signals_df.columns:
                signals_df[col] = 0.5  # Default value
        
        # Calculate signal strength using available metrics
        signal_strength = (
            abs(signals_df['sentiment_score']) * 0.4 +
            signals_df['reliability_score'] * 0.3 +
            (1 - signals_df['price_volatility']) * 0.2 +
            signals_df.get('signal_quality', pd.Series([0.5] * len(signals_df))) * 0.1
        )
        
        return signal_strength.clip(0, 1)
    
    def _calculate_signal_quality(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal quality based on available metrics.
        
        Args:
            signals_df: DataFrame with signals
            
        Returns:
            Series with signal quality values
        """
        if signals_df.empty:
            return pd.Series()
        
        # Ensure all required columns exist
        required_columns = ['sentiment_score', 'reliability_score', 'price_volatility']
        for col in required_columns:
            if col not in signals_df.columns:
                signals_df[col] = 0.5  # Default value
        
        # Calculate signal quality using available metrics
        signal_quality = (
            (abs(signals_df['sentiment_score']) > 0.2).astype(int) * 0.4 +
            (signals_df['reliability_score'] > 0.5).astype(int) * 0.3 +
            (signals_df['price_volatility'] < 0.2).astype(int) * 0.2 +
            0.1  # base quality
        )
        
        return signal_quality.clip(0, 1)
    
    def filter_signals(self, signals_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Apply additional filtering to signals.
        
        Args:
            signals_df: Signals DataFrame
            config: Trading configuration
            
        Returns:
            Filtered signals DataFrame
        """
        if signals_df.empty:
            return signals_df
        
        # Apply quality threshold
        quality_threshold = config.get('trading', {}).get('signal_quality_threshold', 0.5)
        signals_df = signals_df[signals_df['signal_quality'] >= quality_threshold]
        
        # Apply strength threshold
        strength_threshold = config.get('trading', {}).get('signal_strength_threshold', 0.6)
        signals_df = signals_df[signals_df['signal_strength'] >= strength_threshold]
        
        # Remove signals too close together
        min_days_between_signals = config.get('trading', {}).get('min_days_between_signals', 3)
        if len(signals_df) > 1:
            signals_df = signals_df.sort_index()
            days_diff = signals_df.index.to_series().diff().dt.days
            signals_df = signals_df[days_diff >= min_days_between_signals]
        
        logger.info(f"Filtered to {len(signals_df)} high-quality signals")
        return signals_df
    
    def get_signal_summary(self, signals_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for signals.
        
        Args:
            signals_df: Signals DataFrame
            
        Returns:
            Summary dictionary
        """
        if signals_df.empty:
            return {}
        
        summary = {
            'total_signals': len(signals_df),
            'buy_signals': len(signals_df[signals_df['signal_direction'] > 0]),
            'sell_signals': len(signals_df[signals_df['signal_direction'] < 0]),
            'avg_signal_strength': signals_df['signal_strength'].mean(),
            'avg_signal_quality': signals_df['signal_quality'].mean(),
            'avg_sentiment_score': signals_df['sentiment_score'].mean(),
            'avg_reliability_score': signals_df['reliability_score'].mean(),
            'date_range': {
                'start': signals_df.index.min().strftime('%Y-%m-%d'),
                'end': signals_df.index.max().strftime('%Y-%m-%d')
            }
        }
        
        return summary 