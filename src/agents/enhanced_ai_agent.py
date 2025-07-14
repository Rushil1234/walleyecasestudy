"""
Enhanced AI Agent with Advanced NLP and Chain-of-Thought Reasoning

Uses Ollama + Mistral for local LLM reasoning, advanced NLP for novelty detection,
and sophisticated market analysis with memory and learning capabilities.
"""

import ollama
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class EnhancedAIAgent:
    """
    Enhanced AI agent with advanced NLP, novelty detection, and chain-of-thought reasoning.
    """
    
    def __init__(self, model_name: str = "mistral", memory_file: str = "data/enhanced_agent_memory.json", use_llm: bool = True):
        """
        Initialize the enhanced AI agent.
        
        Args:
            model_name: Ollama model name (default: mistral)
            memory_file: File to store agent memory
            use_llm: Whether to use real LLM (Ollama + Mistral)
        """
        self.model_name = model_name
        self.memory_file = Path(memory_file)
        self.use_llm = use_llm
        self.memory = self._load_memory()
        
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        
        # Handle NLTK data loading with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not available, using basic stopwords")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Knowledge base for market analysis
        self.knowledge_base = {
            'historical_events': [],
            'signal_patterns': [],
            'market_regimes': [],
            'entity_relationships': defaultdict(list),
            'novelty_patterns': [],
            'risk_indicators': []
        }
        
        # Test LLM connection if enabled
        if self.use_llm:
            self._test_llm_connection()
    
    def _test_llm_connection(self):
        """Test LLM connection and log status."""
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': 'Hello, this is a test message. Please respond with "OK" if you can read this.'
                }
            ])
            logger.info(f"✅ Enhanced AI Agent LLM connection successful with {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Enhanced AI Agent LLM connection failed: {e}")
            logger.info("Enhanced AI Agent will use rule-based analysis")
    
    def _load_memory(self) -> Dict:
        """
        Load agent memory from file.
        
        Returns:
            Memory dictionary
        """
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading enhanced agent memory: {e}")
        
        return {
            'events': [],
            'signals': [],
            'outcomes': [],
            'patterns': [],
            'novelty_scores': {},
            'entity_mentions': defaultdict(int),
            'market_insights': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_memory(self):
        """Save agent memory to file."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            self.memory['last_updated'] = datetime.now().isoformat()
            
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving enhanced agent memory: {e}")
    
    def analyze_market_with_cot(
        self,
        equity_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame,
        signals: pd.DataFrame,
        trading_results: Dict
    ) -> Dict:
        """
        Analyze market conditions with chain-of-thought reasoning.
        
        Args:
            equity_data: Dictionary of equity DataFrames
            news_data: News DataFrame
            signals: Signals DataFrame
            trading_results: Trading results
            
        Returns:
            Enhanced agent insights dictionary
        """
        insights = {
            'novelty_score': 0.0,
            'confidence': 0.5,
            'recommendations': [],
            'market_regime': 'unknown',
            'risk_assessment': 'medium',
            'pattern_analysis': {},
            'chain_of_thought': '',
            'entity_analysis': {},
            'semantic_insights': {},
            'trading_signals': []
        }
        
        try:
            # Step 1: Chain-of-thought reasoning
            insights['chain_of_thought'] = self._generate_chain_of_thought(
                equity_data, news_data, signals, trading_results
            )
            
            # Step 2: Advanced novelty detection
            insights['novelty_score'] = self._calculate_advanced_novelty_score(news_data, equity_data)
            
            # Step 3: Entity analysis
            insights['entity_analysis'] = self._analyze_entities(news_data)
            
            # Step 4: Semantic insights
            insights['semantic_insights'] = self._extract_semantic_insights(news_data)
            
            # Step 5: Market regime identification
            insights['market_regime'] = self._identify_market_regime(equity_data)
            
            # Step 6: Generate recommendations
            insights['recommendations'] = self._generate_enhanced_recommendations(
                equity_data, news_data, signals, trading_results, insights
            )
            
            # Step 7: Assess confidence
            insights['confidence'] = self._assess_enhanced_confidence(signals, trading_results, insights)
            
            # Step 8: Pattern analysis
            insights['pattern_analysis'] = self._analyze_enhanced_patterns(signals, trading_results)
            
            # Step 9: Trading signals
            insights['trading_signals'] = self._generate_trading_signals(insights)
            
            # Step 10: Update memory
            self._update_enhanced_memory(equity_data, news_data, signals, trading_results, insights)
            
        except Exception as e:
            logger.error(f"Error in enhanced market analysis: {e}")
        
        return insights
    
    def _generate_chain_of_thought(
        self,
        equity_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame,
        signals: pd.DataFrame,
        trading_results: Dict
    ) -> str:
        """
        Generate chain-of-thought reasoning using LLM.
        
        Args:
            equity_data: Equity data
            news_data: News data
            signals: Trading signals
            trading_results: Trading results
            
        Returns:
            Chain-of-thought reasoning text
        """
        try:
            # Prepare market summary
            market_summary = self._create_market_summary(equity_data, news_data, signals, trading_results)
            
            prompt = f"""
            You are an expert financial analyst specializing in oil markets and geopolitical risk.
            
            Analyze the following market data and provide detailed chain-of-thought reasoning:
            
            {market_summary}
            
            Please provide step-by-step reasoning covering:
            1. Current market conditions and trends
            2. Key geopolitical factors affecting oil markets
            3. Sentiment analysis of recent news
            4. Signal quality and reliability assessment
            5. Risk factors and potential scenarios
            6. Trading strategy implications
            
            Format your response as a detailed analysis with clear reasoning steps.
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating chain-of-thought: {e}")
            return "Chain-of-thought analysis unavailable due to technical issues."
    
    def _create_market_summary(
        self,
        equity_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame,
        signals: pd.DataFrame,
        trading_results: Dict
    ) -> str:
        """
        Create a comprehensive market summary for LLM analysis.
        
        Args:
            equity_data: Equity data
            news_data: News data
            signals: Trading signals
            trading_results: Trading results
            
        Returns:
            Market summary text
        """
        summary_parts = []
        
        # Equity data summary
        if equity_data:
            summary_parts.append("EQUITY DATA:")
            for symbol, data in equity_data.items():
                if not data.empty and 'Returns' in data.columns:
                    returns = data['Returns'].dropna()
                    volatility = returns.std() * np.sqrt(252)
                    total_return = (1 + returns).prod() - 1
                    summary_parts.append(f"- {symbol}: Vol={volatility:.2%}, Return={total_return:.2%}")
        
        # News data summary
        if not news_data.empty:
            summary_parts.append(f"\nNEWS DATA:")
            summary_parts.append(f"- Total articles: {len(news_data)}")
            if 'sentiment_score' in news_data.columns:
                avg_sentiment = news_data['sentiment_score'].mean()
                summary_parts.append(f"- Average sentiment: {avg_sentiment:.3f}")
            if 'source' in news_data.columns:
                sources = news_data['source'].value_counts().head(3)
                summary_parts.append(f"- Top sources: {', '.join(sources.index)}")
        
        # Signals summary
        if not signals.empty:
            summary_parts.append(f"\nTRADING SIGNALS:")
            summary_parts.append(f"- Total signals: {len(signals)}")
            buy_signals = len(signals[signals['signal_direction'] > 0])
            sell_signals = len(signals[signals['signal_direction'] < 0])
            summary_parts.append(f"- Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if 'signal_strength' in signals.columns:
                avg_strength = signals['signal_strength'].mean()
                summary_parts.append(f"- Average signal strength: {avg_strength:.3f}")
        
        # Trading results summary
        if trading_results:
            summary_parts.append(f"\nTRADING RESULTS:")
            perf = trading_results.get('performance_summary', {})
            summary_parts.append(f"- Total return: {perf.get('total_return', 0):.2%}")
            summary_parts.append(f"- Sharpe ratio: {perf.get('sharpe_ratio', 0):.3f}")
            summary_parts.append(f"- Max drawdown: {perf.get('max_drawdown', 0):.2%}")
        
        return "\n".join(summary_parts)
    
    def _calculate_advanced_novelty_score(
        self,
        news_data: pd.DataFrame,
        equity_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Calculate advanced novelty score using multiple factors.
        
        Args:
            news_data: News DataFrame
            equity_data: Equity data
            
        Returns:
            Novelty score (0-1)
        """
        if news_data.empty:
            return 0.0
        
        novelty_factors = []
        
        # 1. Content novelty (semantic similarity)
        content_novelty = self._calculate_content_novelty(news_data)
        novelty_factors.append(content_novelty)
        
        # 2. Entity novelty (new entities mentioned)
        entity_novelty = self._calculate_entity_novelty(news_data)
        novelty_factors.append(entity_novelty)
        
        # 3. Sentiment novelty (unusual sentiment patterns)
        sentiment_novelty = self._calculate_sentiment_novelty(news_data)
        novelty_factors.append(sentiment_novelty)
        
        # 4. Price novelty (unusual price movements)
        price_novelty = self._calculate_price_novelty(equity_data)
        novelty_factors.append(price_novelty)
        
        # 5. Source novelty (new or unusual sources)
        source_novelty = self._calculate_source_novelty(news_data)
        novelty_factors.append(source_novelty)
        
        # Combine factors with weights
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Content novelty weighted highest
        novelty_score = sum(factor * weight for factor, weight in zip(novelty_factors, weights))
        
        return min(1.0, novelty_score)
    
    def _calculate_content_novelty(self, news_data: pd.DataFrame) -> float:
        """Calculate content novelty based on semantic similarity."""
        if len(news_data) < 2:
            return 0.5
        
        # Extract recent texts
        recent_texts = []
        for _, row in news_data.tail(20).iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            recent_texts.append(text)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(recent_texts)):
            for j in range(i + 1, len(recent_texts)):
                similarity = self._calculate_text_similarity(recent_texts[i], recent_texts[j])
                similarities.append(similarity)
        
        # Novelty is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1 - avg_similarity
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using spaCy."""
        doc1 = self.nlp(text1.lower())
        doc2 = self.nlp(text2.lower())
        
        # Use spaCy's similarity method
        try:
            return doc1.similarity(doc2)
        except:
            # Fallback to token overlap
            tokens1 = set([token.lemma_ for token in doc1 if not token.is_stop and token.is_alpha])
            tokens2 = set([token.lemma_ for token in doc2 if not token.is_stop and token.is_alpha])
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return intersection / union if union > 0 else 0.0
    
    def _calculate_entity_novelty(self, news_data: pd.DataFrame) -> float:
        """Calculate entity novelty based on new entities mentioned."""
        if news_data.empty:
            return 0.0
        
        # Extract entities from recent news
        recent_entities = set()
        for _, row in news_data.tail(10).iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'ORG', 'PERSON']]
            recent_entities.update(entities)
        
        # Compare with historical entities
        historical_entities = set(self.memory.get('entity_mentions', {}).keys())
        
        if not historical_entities:
            return 0.5  # Default if no historical data
        
        new_entities = recent_entities - historical_entities
        novelty_ratio = len(new_entities) / len(recent_entities) if recent_entities else 0
        
        return min(1.0, novelty_ratio)
    
    def _calculate_sentiment_novelty(self, news_data: pd.DataFrame) -> float:
        """Calculate sentiment novelty based on unusual sentiment patterns."""
        if news_data.empty or 'sentiment_score' not in news_data.columns:
            return 0.0
        
        recent_sentiments = news_data['sentiment_score'].tail(20)
        
        if len(recent_sentiments) < 5:
            return 0.5
        
        # Calculate sentiment volatility
        sentiment_volatility = recent_sentiments.std()
        
        # Calculate sentiment trend
        sentiment_trend = recent_sentiments.diff().mean()
        
        # Novelty based on high volatility or unusual trends
        volatility_novelty = min(1.0, sentiment_volatility * 5)  # Scale volatility
        trend_novelty = min(1.0, abs(sentiment_trend) * 10)  # Scale trend
        
        return (volatility_novelty + trend_novelty) / 2
    
    def _calculate_price_novelty(self, equity_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate price novelty based on unusual price movements."""
        if not equity_data or 'XOP' not in equity_data:
            return 0.0
        
        xop_data = equity_data['XOP']
        if xop_data.empty or 'Returns' not in xop_data.columns:
            return 0.0
        
        returns = xop_data['Returns'].dropna().tail(20)
        
        if len(returns) < 5:
            return 0.5
        
        # Calculate return volatility
        return_volatility = returns.std()
        
        # Calculate unusual returns (beyond 2 standard deviations)
        mean_return = returns.mean()
        std_return = returns.std()
        unusual_returns = returns[abs(returns - mean_return) > 2 * std_return]
        
        volatility_novelty = min(1.0, return_volatility * 20)  # Scale volatility
        unusual_novelty = len(unusual_returns) / len(returns)
        
        return (volatility_novelty + unusual_novelty) / 2
    
    def _calculate_source_novelty(self, news_data: pd.DataFrame) -> float:
        """Calculate source novelty based on new or unusual sources."""
        if news_data.empty or 'source' not in news_data.columns:
            return 0.0
        
        recent_sources = news_data['source'].tail(20).value_counts()
        
        # Check for new sources
        historical_sources = set(self.memory.get('source_patterns', {}).keys())
        new_sources = set(recent_sources.index) - historical_sources
        
        # Calculate source diversity
        source_diversity = len(recent_sources) / len(news_data.tail(20))
        
        new_source_ratio = len(new_sources) / len(recent_sources) if recent_sources.size > 0 else 0
        
        return (source_diversity + new_source_ratio) / 2
    
    def _analyze_entities(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze entities mentioned in news data.
        
        Args:
            news_data: News DataFrame
            
        Returns:
            Entity analysis results
        """
        entity_analysis = {
            'countries': [],
            'organizations': [],
            'people': [],
            'entity_frequency': {},
            'entity_relationships': {},
            'geopolitical_hotspots': []
        }
        
        if news_data.empty:
            return entity_analysis
        
        # Extract entities from all news
        all_entities = defaultdict(list)
        
        for _, row in news_data.iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == 'GPE':  # Countries, cities
                    all_entities['countries'].append(ent.text)
                elif ent.label_ == 'ORG':  # Organizations
                    all_entities['organizations'].append(ent.text)
                elif ent.label_ == 'PERSON':  # People
                    all_entities['people'].append(ent.text)
        
        # Calculate entity frequencies
        for entity_type, entities in all_entities.items():
            entity_counts = Counter(entities)
            entity_analysis[f'{entity_type}_frequency'] = dict(entity_counts.most_common(10))
        
        # Identify geopolitical hotspots (frequently mentioned countries)
        country_counts = Counter(all_entities['countries'])
        entity_analysis['geopolitical_hotspots'] = [
            country for country, count in country_counts.most_common(5)
            if count >= 3  # Mentioned at least 3 times
        ]
        
        return entity_analysis
    
    def _extract_semantic_insights(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract semantic insights from news data.
        
        Args:
            news_data: News DataFrame
            
        Returns:
            Semantic insights
        """
        insights = {
            'key_themes': [],
            'sentiment_trends': {},
            'topic_clusters': [],
            'semantic_patterns': {}
        }
        
        if news_data.empty:
            return insights
        
        # Extract key themes using spaCy
        all_texts = []
        for _, row in news_data.iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            all_texts.append(text)
        
        # Find common noun phrases and entities
        common_phrases = self._extract_common_phrases(all_texts)
        insights['key_themes'] = common_phrases[:10]
        
        # Analyze sentiment trends over time
        if 'sentiment_score' in news_data.columns and 'published_date' in news_data.columns:
            sentiment_trends = self._analyze_sentiment_trends(news_data)
            insights['sentiment_trends'] = sentiment_trends
        
        return insights
    
    def _extract_common_phrases(self, texts: List[str]) -> List[str]:
        """Extract common phrases from texts."""
        phrase_counts = Counter()
        
        for text in texts:
            doc = self.nlp(text)
            
            # Extract noun chunks
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word phrases
                    phrase_counts[chunk.text.lower()] += 1
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'ORG', 'PERSON']:
                    phrase_counts[ent.text.lower()] += 1
        
        return [phrase for phrase, count in phrase_counts.most_common(20)]
    
    def _analyze_sentiment_trends(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        # Group by date and calculate average sentiment
        daily_sentiment = news_data.groupby('published_date')['sentiment_score'].agg(['mean', 'std', 'count'])
        
        return {
            'daily_averages': daily_sentiment['mean'].to_dict(),
            'sentiment_volatility': daily_sentiment['std'].to_dict(),
            'article_counts': daily_sentiment['count'].to_dict()
        }
    
    def _identify_market_regime(self, equity_data: Dict[str, pd.DataFrame]) -> str:
        """
        Identify current market regime.
        
        Args:
            equity_data: Equity data
            
        Returns:
            Market regime classification
        """
        if not equity_data or 'SPY' not in equity_data:
            return 'unknown'
        
        spy_data = equity_data['SPY']
        if spy_data.empty or 'Returns' not in spy_data.columns:
            return 'unknown'
        
        returns = spy_data['Returns'].dropna().tail(30)  # Last 30 days
        
        if len(returns) < 10:
            return 'unknown'
        
        # Calculate regime indicators
        volatility = returns.std() * np.sqrt(252)
        mean_return = returns.mean() * 252
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Classify regime
        if volatility > 0.25:  # High volatility
            return 'high_volatility'
        elif sharpe_ratio > 1.0:  # Good risk-adjusted returns
            return 'bull_market'
        elif sharpe_ratio < -0.5:  # Poor risk-adjusted returns
            return 'bear_market'
        else:
            return 'sideways_market'
    
    def _generate_enhanced_recommendations(
        self,
        equity_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame,
        signals: pd.DataFrame,
        trading_results: Dict,
        insights: Dict
    ) -> List[str]:
        """
        Generate enhanced recommendations using LLM.
        
        Args:
            equity_data: Equity data
            news_data: News data
            signals: Trading signals
            trading_results: Trading results
            insights: Current insights
            
        Returns:
            List of recommendations
        """
        try:
            # Create recommendation prompt
            prompt = f"""
            Based on the following market analysis, provide 5 specific, actionable trading recommendations:
            
            Market Regime: {insights['market_regime']}
            Novelty Score: {insights['novelty_score']:.3f}
            Confidence: {insights['confidence']:.3f}
            
            Key Insights:
            - Entity Analysis: {insights.get('entity_analysis', {})}
            - Semantic Insights: {insights.get('semantic_insights', {})}
            - Pattern Analysis: {insights.get('pattern_analysis', {})}
            
            Current Performance:
            - Total Return: {trading_results.get('performance_summary', {}).get('total_return', 0):.2%}
            - Sharpe Ratio: {trading_results.get('performance_summary', {}).get('sharpe_ratio', 0):.3f}
            
            Please provide 5 specific recommendations in this format:
            1. [Recommendation 1]
            2. [Recommendation 2]
            3. [Recommendation 3]
            4. [Recommendation 4]
            5. [Recommendation 5]
            
            Focus on actionable insights for oil market trading based on geopolitical news analysis.
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            # Parse recommendations
            content = response['message']['content']
            recommendations = []
            
            # Extract numbered recommendations
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                    recommendation = line.strip()[2:].strip()
                    if recommendation:
                        recommendations.append(recommendation)
            
            return recommendations[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error generating enhanced recommendations: {e}")
            return [
                "Monitor geopolitical developments closely",
                "Adjust position sizing based on novelty scores",
                "Consider hedging strategies for high-risk periods",
                "Review signal quality and reliability metrics",
                "Maintain diversified exposure across energy assets"
            ]
    
    def _assess_enhanced_confidence(
        self,
        signals: pd.DataFrame,
        trading_results: Dict,
        insights: Dict
    ) -> float:
        """
        Assess confidence level based on multiple factors.
        
        Args:
            signals: Trading signals
            trading_results: Trading results
            insights: Current insights
            
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        # Signal quality
        if not signals.empty and 'signal_quality' in signals.columns:
            avg_quality = signals['signal_quality'].mean()
            confidence_factors.append(avg_quality)
        
        # Historical performance
        if trading_results and 'performance_summary' in trading_results:
            perf = trading_results['performance_summary']
            sharpe = perf.get('sharpe_ratio', 0)
            sharpe_confidence = min(1.0, max(0.0, sharpe / 2.0))  # Normalize Sharpe
            confidence_factors.append(sharpe_confidence)
        
        # Novelty score (lower novelty = higher confidence)
        novelty_confidence = 1 - insights.get('novelty_score', 0.5)
        confidence_factors.append(novelty_confidence)
        
        # Market regime stability
        regime = insights.get('market_regime', 'unknown')
        regime_confidence = 0.8 if regime in ['bull_market', 'bear_market'] else 0.6
        confidence_factors.append(regime_confidence)
        
        # Entity analysis confidence
        entity_analysis = insights.get('entity_analysis', {})
        entity_confidence = 0.7 if entity_analysis else 0.5
        confidence_factors.append(entity_confidence)
        
        # Calculate weighted average
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        return min(1.0, max(0.0, confidence))
    
    def _analyze_enhanced_patterns(
        self,
        signals: pd.DataFrame,
        trading_results: Dict
    ) -> Dict[str, Any]:
        """
        Analyze enhanced patterns in signals and trading results.
        
        Args:
            signals: Trading signals
            trading_results: Trading results
            
        Returns:
            Pattern analysis results
        """
        patterns = {
            'signal_patterns': {},
            'performance_patterns': {},
            'timing_patterns': {},
            'correlation_patterns': {}
        }
        
        if not signals.empty:
            # Signal timing patterns
            if 'signal_direction' in signals.columns:
                buy_signals = signals[signals['signal_direction'] > 0]
                sell_signals = signals[signals['signal_direction'] < 0]
                
                patterns['signal_patterns'] = {
                    'buy_signal_count': len(buy_signals),
                    'sell_signal_count': len(sell_signals),
                    'buy_sell_ratio': len(buy_signals) / len(sell_signals) if len(sell_signals) > 0 else float('inf')
                }
        
        if trading_results and 'performance_summary' in trading_results:
            perf = trading_results['performance_summary']
            patterns['performance_patterns'] = {
                'total_return': perf.get('total_return', 0),
                'sharpe_ratio': perf.get('sharpe_ratio', 0),
                'max_drawdown': perf.get('max_drawdown', 0),
                'win_rate': perf.get('win_rate', 0)
            }
        
        return patterns
    
    def _generate_trading_signals(self, insights: Dict) -> List[Dict]:
        """
        Generate trading signals based on insights.
        
        Args:
            insights: Current insights
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Generate signals based on novelty and confidence
        novelty_score = insights.get('novelty_score', 0)
        confidence = insights.get('confidence', 0)
        market_regime = insights.get('market_regime', 'unknown')
        
        # High novelty, high confidence = strong signal
        if novelty_score > 0.7 and confidence > 0.7:
            signals.append({
                'type': 'strong_buy',
                'reason': f'High novelty ({novelty_score:.2f}) and confidence ({confidence:.2f})',
                'strength': 0.9
            })
        
        # High novelty, low confidence = cautious signal
        elif novelty_score > 0.7 and confidence < 0.5:
            signals.append({
                'type': 'cautious_buy',
                'reason': f'High novelty but low confidence - monitor closely',
                'strength': 0.6
            })
        
        # Low novelty, high confidence = stable signal
        elif novelty_score < 0.3 and confidence > 0.7:
            signals.append({
                'type': 'stable_hold',
                'reason': f'Low novelty, stable conditions - maintain positions',
                'strength': 0.7
            })
        
        # Bear market signals
        if market_regime == 'bear_market':
            signals.append({
                'type': 'risk_reduction',
                'reason': 'Bear market detected - consider reducing exposure',
                'strength': 0.8
            })
        
        return signals
    
    def _update_enhanced_memory(
        self,
        equity_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame,
        signals: pd.DataFrame,
        trading_results: Dict,
        insights: Dict
    ):
        """
        Update enhanced memory with new information.
        
        Args:
            equity_data: Equity data
            news_data: News data
            signals: Trading signals
            trading_results: Trading results
            insights: Current insights
        """
        # Update entity mentions
        entity_analysis = insights.get('entity_analysis', {})
        for entity_type, entities in entity_analysis.items():
            if 'frequency' in entity_type:
                for entity, count in entities.items():
                    self.memory['entity_mentions'][entity] += count
        
        # Update patterns
        self.memory['patterns'].append({
            'timestamp': datetime.now().isoformat(),
            'novelty_score': insights.get('novelty_score', 0),
            'market_regime': insights.get('market_regime', 'unknown'),
            'confidence': insights.get('confidence', 0)
        })
        
        # Update insights
        self.memory['market_insights'].append({
            'timestamp': datetime.now().isoformat(),
            'insights': insights
        })
        
        # Keep only recent data
        max_entries = 1000
        if len(self.memory['patterns']) > max_entries:
            self.memory['patterns'] = self.memory['patterns'][-max_entries:]
        if len(self.memory['market_insights']) > max_entries:
            self.memory['market_insights'] = self.memory['market_insights'][-max_entries:]
        
        # Save memory (handle timestamp serialization)
        try:
            self._save_memory()
        except Exception as e:
            logger.warning(f"Could not save memory: {e}")
    
    def get_enhanced_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of enhanced memory.
        
        Returns:
            Memory summary
        """
        summary = {
            'total_events': len(self.memory.get('events', [])),
            'total_patterns': len(self.memory.get('patterns', [])),
            'total_insights': len(self.memory.get('market_insights', [])),
            'entity_count': len(self.memory.get('entity_mentions', {})),
            'last_updated': self.memory.get('last_updated', 'unknown'),
            'top_entities': dict(sorted(
                self.memory.get('entity_mentions', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }
        
        return summary 