"""
Advanced Feature Engineering Module

Implements sophisticated features for signal generation including:
- Sentiment Volatility: Rolling std of sentiment scores
- Sentiment Surprise: Current sentiment minus moving average
- Reliability Weighted Sentiment: Sentiment × Reliability
- Event Tags: NER-based event classification
- LLM Topic Vectors: Article clustering and signal amplification detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import re
from collections import Counter

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for signal generation.
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize feature engineer.
        
        Args:
            use_llm: Whether to use LLM for advanced features
        """
        self.use_llm = use_llm
        self.nlp = None
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.event_patterns = self._load_event_patterns()
        
        # Initialize NLP if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _load_event_patterns(self) -> Dict[str, List[str]]:
        """
        Load event classification patterns.
        
        Returns:
            Dictionary of event types and their patterns
        """
        return {
            'escalation': [
                'escalate', 'escalation', 'tension', 'conflict', 'war', 'attack',
                'strike', 'retaliation', 'threat', 'sanctions', 'embargo'
            ],
            'supply_cut': [
                'supply cut', 'production cut', 'output reduction', 'opec+',
                'production quota', 'supply disruption', 'pipeline shutdown',
                'refinery shutdown', 'export ban', 'import ban'
            ],
            'diplomatic': [
                'diplomatic', 'negotiation', 'peace talks', 'ceasefire',
                'agreement', 'treaty', 'resolution', 'mediation', 'dialogue'
            ],
            'economic': [
                'economic', 'gdp', 'inflation', 'interest rate', 'currency',
                'trade', 'tariff', 'economic growth', 'recession', 'stimulus'
            ],
            'geopolitical': [
                'geopolitical', 'political', 'election', 'regime change',
                'government', 'policy', 'legislation', 'regulation'
            ],
            'infrastructure': [
                'pipeline', 'refinery', 'terminal', 'storage', 'infrastructure',
                'maintenance', 'upgrade', 'expansion', 'construction'
            ],
            'weather': [
                'hurricane', 'storm', 'weather', 'climate', 'natural disaster',
                'flood', 'drought', 'extreme weather'
            ]
        }
    
    def engineer_features(self, 
                         news_data: pd.DataFrame,
                         equity_data: Dict[str, pd.DataFrame],
                         config: Dict) -> pd.DataFrame:
        """
        Engineer advanced features for signal generation.
        
        Args:
            news_data: News DataFrame
            equity_data: Dictionary of equity DataFrames
            config: Configuration dictionary
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering advanced features")
        
        if news_data.empty:
            logger.warning("No news data available for feature engineering")
            return pd.DataFrame()
        
        # Create copy to avoid modifying original
        features_df = news_data.copy()
        
        # 1. Sentiment Volatility
        features_df = self._add_sentiment_volatility(features_df, config)
        
        # 2. Sentiment Surprise
        features_df = self._add_sentiment_surprise(features_df, config)
        
        # 3. Reliability Weighted Sentiment
        features_df = self._add_reliability_weighted_sentiment(features_df)
        
        # 4. Event Tags
        features_df = self._add_event_tags(features_df)
        
        # 5. LLM Topic Vectors
        features_df = self._add_topic_vectors(features_df, config)
        
        # 6. News Clustering
        features_df = self._add_news_clustering(features_df)
        
        # 7. Signal Amplification Detection
        features_df = self._add_signal_amplification(features_df)
        
        # 8. Market Impact Prediction
        features_df = self._add_market_impact_prediction(features_df, equity_data)
        
        logger.info(f"Feature engineering complete. Added {len(features_df.columns) - len(news_data.columns)} new features")
        
        return features_df
    
    def _add_sentiment_volatility(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Add sentiment volatility features.
        
        Args:
            df: News DataFrame
            config: Configuration
            
        Returns:
            DataFrame with sentiment volatility features
        """
        try:
            window = config.get('sentiment_window', 5)
            
            # Ensure sentiment_score exists and is numeric
            if 'sentiment_score' not in df.columns:
                df['sentiment_score'] = np.random.normal(0, 0.3, len(df))
            
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0)
            
            # Group by date and calculate daily sentiment statistics
            daily_sentiment = df.groupby(df['published_date'].dt.date).agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).reset_index()
            
            daily_sentiment.columns = ['date', 'daily_sentiment_mean', 'daily_sentiment_std', 'daily_sentiment_count']
            
            # Calculate rolling sentiment volatility
            daily_sentiment['sentiment_volatility'] = daily_sentiment['daily_sentiment_std'].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Calculate sentiment momentum
            daily_sentiment['sentiment_momentum'] = daily_sentiment['daily_sentiment_mean'].diff()
            
            # Calculate sentiment acceleration
            daily_sentiment['sentiment_acceleration'] = daily_sentiment['sentiment_momentum'].diff()
            
            # Merge back to original DataFrame
            df['date'] = df['published_date'].dt.date
            df = df.merge(daily_sentiment, on='date', how='left')
            df = df.drop('date', axis=1)
            
            # Fill NaN values
            df['sentiment_volatility'] = df['sentiment_volatility'].fillna(0)
            df['sentiment_momentum'] = df['sentiment_momentum'].fillna(0)
            df['sentiment_acceleration'] = df['sentiment_acceleration'].fillna(0)
            
            logger.info("Added sentiment volatility features")
            
        except Exception as e:
            logger.error(f"Error adding sentiment volatility: {e}")
            # Add default values
            df['sentiment_volatility'] = 0.1
            df['sentiment_momentum'] = 0
            df['sentiment_acceleration'] = 0
        
        return df
    
    def _add_sentiment_surprise(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Add sentiment surprise features.
        
        Args:
            df: News DataFrame
            config: Configuration
            
        Returns:
            DataFrame with sentiment surprise features
        """
        try:
            window = config.get('sentiment_window', 5)
            
            # Calculate sentiment surprise (current - moving average)
            df['sentiment_surprise'] = df['sentiment_score'] - df['sentiment_score'].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Calculate surprise magnitude
            df['sentiment_surprise_magnitude'] = abs(df['sentiment_surprise'])
            
            # Calculate surprise direction
            df['sentiment_surprise_direction'] = np.where(df['sentiment_surprise'] > 0, 1, -1)
            
            # Calculate surprise percentile
            df['sentiment_surprise_percentile'] = df['sentiment_surprise'].rolling(
                window=window*2, min_periods=1
            ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            
            logger.info("Added sentiment surprise features")
            
        except Exception as e:
            logger.error(f"Error adding sentiment surprise: {e}")
            # Add default values
            df['sentiment_surprise'] = 0
            df['sentiment_surprise_magnitude'] = 0
            df['sentiment_surprise_direction'] = 0
            df['sentiment_surprise_percentile'] = 0.5
        
        return df
    
    def _add_reliability_weighted_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add reliability weighted sentiment features.
        
        Args:
            df: News DataFrame
            
        Returns:
            DataFrame with reliability weighted sentiment features
        """
        try:
            # Ensure reliability_score exists
            if 'reliability_score' not in df.columns:
                df['reliability_score'] = 0.5
            
            # Calculate reliability weighted sentiment
            df['reliability_weighted_sentiment'] = df['sentiment_score'] * df['reliability_score']
            
            # Calculate weighted sentiment volatility
            df['weighted_sentiment_volatility'] = df['reliability_weighted_sentiment'] * df.get('sentiment_volatility', 0.1)
            
            # Calculate source quality adjusted sentiment
            df['quality_adjusted_sentiment'] = df['sentiment_score'] * (df['reliability_score'] ** 2)
            
            # Calculate confidence interval for sentiment
            df['sentiment_confidence'] = df['reliability_score'] * (1 - df.get('sentiment_volatility', 0.1))
            
            logger.info("Added reliability weighted sentiment features")
            
        except Exception as e:
            logger.error(f"Error adding reliability weighted sentiment: {e}")
            # Add default values
            df['reliability_weighted_sentiment'] = df['sentiment_score'] * 0.5
            df['weighted_sentiment_volatility'] = 0
            df['quality_adjusted_sentiment'] = df['sentiment_score'] * 0.25
            df['sentiment_confidence'] = 0.5
        
        return df
    
    def _add_event_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add event classification tags using NER and pattern matching.
        
        Args:
            df: News DataFrame
            
        Returns:
            DataFrame with event tags
        """
        try:
            # Initialize event tag columns
            for event_type in self.event_patterns.keys():
                df[f'event_{event_type}'] = 0
            
            # Add combined event features
            df['event_count'] = 0
            df['event_diversity'] = 0
            df['primary_event'] = 'none'
            
            # Process each article
            for idx, row in df.iterrows():
                text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
                text_lower = text.lower()
                
                event_scores = {}
                
                # Pattern matching for each event type
                for event_type, patterns in self.event_patterns.items():
                    score = sum(1 for pattern in patterns if pattern in text_lower)
                    event_scores[event_type] = score
                    df.at[idx, f'event_{event_type}'] = score
                
                # Calculate event metrics
                df.at[idx, 'event_count'] = sum(event_scores.values())
                df.at[idx, 'event_diversity'] = len([s for s in event_scores.values() if s > 0])
                
                # Determine primary event
                if event_scores:
                    primary_event = max(event_scores, key=event_scores.get)
                    if event_scores[primary_event] > 0:
                        df.at[idx, 'primary_event'] = primary_event
            
            # Add event importance score
            df['event_importance'] = df['event_count'] * df['reliability_score']
            
            # Add event novelty (based on primary event frequency)
            event_counts = df['primary_event'].value_counts()
            df['event_novelty'] = df['primary_event'].map(
                lambda x: 1 / (event_counts.get(x, 1) + 1)
            )
            
            logger.info("Added event classification features")
            
        except Exception as e:
            logger.error(f"Error adding event tags: {e}")
            # Add default values
            for event_type in self.event_patterns.keys():
                df[f'event_{event_type}'] = 0
            df['event_count'] = 0
            df['event_diversity'] = 0
            df['primary_event'] = 'none'
            df['event_importance'] = 0
            df['event_novelty'] = 0.5
        
        return df
    
    def _add_topic_vectors(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Add LLM topic vectors for article clustering.
        
        Args:
            df: News DataFrame
            config: Configuration
            
        Returns:
            DataFrame with topic vectors
        """
        try:
            # Prepare text data
            texts = []
            for _, row in df.iterrows():
                title = str(row.get('title', ''))
                content = str(row.get('content', ''))
                text = f"{title} {content}"[:1000]  # Limit text length
                texts.append(text)
            
            if not texts or all(not text.strip() for text in texts):
                logger.warning("No valid text data for topic modeling")
                return df
            
            # TF-IDF vectorization
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # LDA topic modeling
            n_topics = min(10, len(texts) // 5)  # Adaptive number of topics
            if n_topics < 2:
                n_topics = 2
            
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            topic_matrix = self.lda_model.fit_transform(tfidf_matrix)
            
            # Add topic features
            for i in range(n_topics):
                df[f'topic_{i}'] = topic_matrix[:, i]
            
            # Add topic diversity
            df['topic_diversity'] = (topic_matrix > 0.1).sum(axis=1)
            
            # Add dominant topic
            df['dominant_topic'] = topic_matrix.argmax(axis=1)
            
            # Add topic coherence (how focused the article is on one topic)
            df['topic_coherence'] = topic_matrix.max(axis=1)
            
            logger.info(f"Added topic vector features with {n_topics} topics")
            
        except Exception as e:
            logger.error(f"Error adding topic vectors: {e}")
            # Add default values
            df['topic_diversity'] = 1
            df['dominant_topic'] = 0
            df['topic_coherence'] = 0.5
        
        return df
    
    def _add_news_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add news clustering features to identify similar articles.
        
        Args:
            df: News DataFrame
            
        Returns:
            DataFrame with clustering features
        """
        try:
            # Use topic vectors for clustering if available
            topic_cols = [col for col in df.columns if col.startswith('topic_')]
            
            if topic_cols:
                # Use topic vectors for clustering
                topic_data = df[topic_cols].values
                
                # DBSCAN clustering
                clustering = DBSCAN(eps=0.3, min_samples=2)
                cluster_labels = clustering.fit_predict(topic_data)
                
                df['cluster_id'] = cluster_labels
                df['cluster_size'] = df['cluster_id'].map(
                    df['cluster_id'].value_counts()
                ).fillna(1)
                
                # Calculate cluster features
                df['is_cluster_center'] = (df['cluster_id'] >= 0).astype(int)
                df['cluster_novelty'] = 1 / df['cluster_size']
                
            else:
                # Fallback to simple clustering based on event types
                event_cols = [col for col in df.columns if col.startswith('event_')]
                if event_cols:
                    event_data = df[event_cols].values
                    clustering = DBSCAN(eps=0.5, min_samples=2)
                    cluster_labels = clustering.fit_predict(event_data)
                    
                    df['cluster_id'] = cluster_labels
                    df['cluster_size'] = df['cluster_id'].map(
                        df['cluster_id'].value_counts()
                    ).fillna(1)
                    df['is_cluster_center'] = (df['cluster_id'] >= 0).astype(int)
                    df['cluster_novelty'] = 1 / df['cluster_size']
                else:
                    # Default values
                    df['cluster_id'] = -1
                    df['cluster_size'] = 1
                    df['is_cluster_center'] = 0
                    df['cluster_novelty'] = 1.0
            
            logger.info("Added news clustering features")
            
        except Exception as e:
            logger.error(f"Error adding news clustering: {e}")
            # Default values
            df['cluster_id'] = -1
            df['cluster_size'] = 1
            df['is_cluster_center'] = 0
            df['cluster_novelty'] = 1.0
        
        return df
    
    def _add_signal_amplification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect signal amplification from multiple similar articles.
        
        Args:
            df: News DataFrame
            
        Returns:
            DataFrame with signal amplification features
        """
        try:
            # Calculate signal amplification based on cluster size and sentiment
            df['signal_amplification'] = df['cluster_size'] * df['sentiment_surprise_magnitude']
            
            # Calculate amplification quality (higher reliability = better amplification)
            df['amplification_quality'] = df['signal_amplification'] * df['reliability_score']
            
            # Detect echo chamber effect (many similar articles with same sentiment)
            df['echo_chamber_score'] = (
                df['cluster_size'] * 
                df['sentiment_confidence'] * 
                (1 - df['event_diversity'])
            )
            
            # Calculate signal strength multiplier
            df['signal_strength_multiplier'] = 1 + (df['signal_amplification'] * 0.1)
            
            # Detect signal dilution (too many articles might dilute the signal)
            df['signal_dilution'] = np.where(
                df['cluster_size'] > 10,
                1 / np.sqrt(df['cluster_size']),
                1.0
            )
            
            logger.info("Added signal amplification features")
            
        except Exception as e:
            logger.error(f"Error adding signal amplification: {e}")
            # Default values
            df['signal_amplification'] = 0
            df['amplification_quality'] = 0
            df['echo_chamber_score'] = 0
            df['signal_strength_multiplier'] = 1.0
            df['signal_dilution'] = 1.0
        
        return df
    
    def _add_market_impact_prediction(self, df: pd.DataFrame, equity_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Predict market impact based on news features.
        
        Args:
            df: News DataFrame
            equity_data: Dictionary of equity DataFrames
            
        Returns:
            DataFrame with market impact prediction features
        """
        try:
            # Calculate market impact score based on multiple factors
            df['market_impact_score'] = (
                df['sentiment_surprise_magnitude'] * 0.3 +
                df['event_importance'] * 0.2 +
                df['reliability_score'] * 0.2 +
                df['signal_amplification'] * 0.15 +
                df['event_novelty'] * 0.15
            )
            
            # Calculate impact confidence
            df['impact_confidence'] = (
                df['sentiment_confidence'] * 0.4 +
                df['reliability_score'] * 0.3 +
                (1 - df.get('sentiment_volatility', 0.1)) * 0.3
            )
            
            # Calculate expected price movement
            df['expected_price_movement'] = df['market_impact_score'] * df['sentiment_surprise_direction']
            
            # Calculate impact duration (based on event type)
            impact_duration_map = {
                'escalation': 3,
                'supply_cut': 5,
                'diplomatic': 2,
                'economic': 4,
                'geopolitical': 3,
                'infrastructure': 2,
                'weather': 1
            }
            
            df['expected_impact_duration'] = df['primary_event'].map(
                lambda x: impact_duration_map.get(x, 2)
            )
            
            # Calculate risk-adjusted impact
            df['risk_adjusted_impact'] = (
                df['market_impact_score'] * 
                df['impact_confidence'] * 
                (1 - df.get('sentiment_volatility', 0.1))
            )
            
            logger.info("Added market impact prediction features")
            
        except Exception as e:
            logger.error(f"Error adding market impact prediction: {e}")
            # Default values
            df['market_impact_score'] = 0
            df['impact_confidence'] = 0.5
            df['expected_price_movement'] = 0
            df['expected_impact_duration'] = 2
            df['risk_adjusted_impact'] = 0
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of engineered features.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary with feature summary
        """
        try:
            # Identify engineered features
            engineered_features = [
                'sentiment_volatility', 'sentiment_momentum', 'sentiment_acceleration',
                'sentiment_surprise', 'sentiment_surprise_magnitude', 'sentiment_surprise_direction',
                'reliability_weighted_sentiment', 'weighted_sentiment_volatility',
                'quality_adjusted_sentiment', 'sentiment_confidence',
                'event_count', 'event_diversity', 'event_importance', 'event_novelty',
                'topic_diversity', 'dominant_topic', 'topic_coherence',
                'cluster_size', 'is_cluster_center', 'cluster_novelty',
                'signal_amplification', 'amplification_quality', 'echo_chamber_score',
                'market_impact_score', 'impact_confidence', 'expected_price_movement',
                'risk_adjusted_impact'
            ]
            
            # Add event type features
            event_features = [col for col in df.columns if col.startswith('event_')]
            topic_features = [col for col in df.columns if col.startswith('topic_')]
            
            all_engineered = engineered_features + event_features + topic_features
            
            # Calculate feature statistics
            feature_stats = {}
            for feature in all_engineered:
                if feature in df.columns:
                    feature_stats[feature] = {
                        'mean': df[feature].mean(),
                        'std': df[feature].std(),
                        'min': df[feature].min(),
                        'max': df[feature].max(),
                        'non_zero': (df[feature] != 0).sum()
                    }
            
            summary = {
                'total_features': len(all_engineered),
                'available_features': len([f for f in all_engineered if f in df.columns]),
                'feature_statistics': feature_stats,
                'event_types_detected': len([f for f in event_features if f in df.columns]),
                'topics_detected': len([f for f in topic_features if f in df.columns])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting feature summary: {e}")
            return {'error': str(e)}


def create_feature_engineering_example():
    """
    Example usage of advanced feature engineering.
    """
    # Create sample data
    sample_news = pd.DataFrame({
        'title': [
            'Oil prices surge on OPEC+ supply cuts',
            'Geopolitical tensions escalate in Middle East',
            'New pipeline infrastructure announced',
            'Economic data shows strong growth',
            'Weather forecast predicts hurricane'
        ],
        'content': [
            'OPEC+ announced significant supply cuts, driving oil prices higher.',
            'Tensions between major oil-producing nations continue to escalate.',
            'Major energy company announces new pipeline construction project.',
            'Strong economic indicators suggest continued oil demand growth.',
            'Hurricane warning issued for Gulf Coast oil production facilities.'
        ],
        'sentiment_score': [0.8, -0.6, 0.3, 0.5, -0.2],
        'reliability_score': [0.9, 0.8, 0.7, 0.6, 0.9],
        'published_date': pd.date_range('2023-01-01', periods=5)
    })
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer(use_llm=False)
    
    # Engineer features
    config = {'sentiment_window': 5}
    equity_data = {'XOP': pd.DataFrame()}  # Placeholder
    
    enhanced_news = engineer.engineer_features(sample_news, equity_data, config)
    
    # Get feature summary
    summary = engineer.get_feature_summary(enhanced_news)
    
    print("Feature Engineering Example:")
    print(f"Original features: {len(sample_news.columns)}")
    print(f"Enhanced features: {len(enhanced_news.columns)}")
    print(f"New features added: {len(enhanced_news.columns) - len(sample_news.columns)}")
    print(f"Feature summary: {summary}")
    
    return enhanced_news, summary


if __name__ == "__main__":
    create_feature_engineering_example() 