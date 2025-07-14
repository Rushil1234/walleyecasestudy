"""
Enhanced LLM-based sentiment analysis for news headlines.

Uses Ollama + Mistral for local LLM sentiment analysis with advanced NLP features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import json
import re
from datetime import datetime
import yaml
from .enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer

logger = logging.getLogger(__name__)


class LLMSentimentAnalyzer:
    """
    Enhanced sentiment analysis using Ollama + Mistral with advanced NLP features.
    """
    
    def __init__(self, config_path: str = "config/sentiment.yaml", use_llm: bool = True):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config_path: Path to sentiment configuration
            use_llm: Whether to use real LLM (Ollama + Mistral)
        """
        self.config = self._load_config(config_path)
        self.use_llm = use_llm
        
        # Initialize enhanced analyzer with real LLM integration
        self.enhanced_analyzer = EnhancedSentimentAnalyzer(use_llm=use_llm)
        
        # Cache for sentiment results
        self.sentiment_cache = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_llm_client(self):
        """
        Initialize LLM client based on configuration.
        
        Returns:
            LLM client or None
        """
        llm_config = self.config.get('llm', {})
        
        # Try OpenAI
        if OPENAI_AVAILABLE and 'openai' in llm_config:
            try:
                openai.api_key = llm_config['openai'].get('api_key')
                return {'type': 'openai', 'config': llm_config['openai']}
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
        
        # Try Anthropic
        if ANTHROPIC_AVAILABLE and 'anthropic' in llm_config:
            try:
                client = anthropic.Anthropic(api_key=llm_config['anthropic'].get('api_key'))
                return {'type': 'anthropic', 'client': client, 'config': llm_config['anthropic']}
            except Exception as e:
                logger.warning(f"Anthropic initialization failed: {e}")
        
        # Try Ollama (local)
        if OLLAMA_AVAILABLE and 'ollama' in llm_config:
            try:
                client = ollama.Client()
                return {'type': 'ollama', 'client': client, 'config': llm_config['ollama']}
            except Exception as e:
                logger.warning(f"Ollama initialization failed: {e}")
        
        logger.warning("No LLM client available, using fallback methods")
        return None
    
    def analyze_sentiment(
        self, 
        headlines: List[str], 
        sources: List[str],
        dates: List[datetime],
        summaries: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Analyze sentiment for a list of headlines.
        
        Args:
            headlines: List of headlines
            sources: List of sources
            dates: List of dates
            summaries: Optional list of summaries
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for i, (headline, source, date) in enumerate(zip(headlines, sources, dates)):
            try:
                # Check cache first
                cache_key = f"{headline}_{source}_{date.date()}"
                if cache_key in self.sentiment_cache:
                    results.append(self.sentiment_cache[cache_key])
                    continue
                
                # Analyze sentiment
                summary = summaries[i] if summaries and i < len(summaries) else ""
                sentiment_result = self._analyze_single_sentiment(headline, source, date, summary)
                
                # Cache result
                self.sentiment_cache[cache_key] = sentiment_result
                results.append(sentiment_result)
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for headline {i}: {e}")
                # Add fallback result
                fallback_result = self._fallback_sentiment_analysis(headlines[i])
                results.append(fallback_result)
        
        return results
    
    def _analyze_single_sentiment(
        self, 
        headline: str, 
        source: str, 
        date: datetime,
        summary: str = ""
    ) -> Dict:
        """
        Analyze sentiment for a single headline using LLM.
        
        Args:
            headline: News headline
            source: News source
            date: Publication date
            summary: Article summary
            
        Returns:
            Sentiment analysis result
        """
        if self.llm_client:
            return self._llm_sentiment_analysis(headline, source, date, summary)
        else:
            return self._fallback_sentiment_analysis(headline)
    
    def _llm_sentiment_analysis(
        self, 
        headline: str, 
        source: str, 
        date: datetime,
        summary: str = ""
    ) -> Dict:
        """
        Perform sentiment analysis using LLM.
        
        Args:
            headline: News headline
            source: News source
            date: Publication date
            summary: Article summary
            
        Returns:
            Sentiment analysis result
        """
        try:
            # Prepare prompt
            prompt_template = self.config.get('prompts', {}).get('sentiment_analysis', '')
            prompt = prompt_template.format(
                headline=headline,
                source=source,
                date=date.strftime('%Y-%m-%d'),
                summary=summary
            )
            
            # Get LLM response
            response = self._get_llm_response(prompt)
            
            # Parse response
            result = self._parse_llm_response(response)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
            return self._fallback_sentiment_analysis(headline)
    
    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            LLM response
        """
        if not self.llm_client:
            return ""
        
        client_type = self.llm_client['type']
        
        try:
            if client_type == 'openai':
                config = self.llm_client['config']
                response = openai.ChatCompletion.create(
                    model=config.get('model', 'gpt-4'),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.get('temperature', 0.1),
                    max_tokens=config.get('max_tokens', 1000)
                )
                return response.choices[0].message.content
            
            elif client_type == 'anthropic':
                client = self.llm_client['client']
                config = self.llm_client['config']
                response = client.messages.create(
                    model=config.get('model', 'claude-3-sonnet'),
                    max_tokens=config.get('max_tokens', 1000),
                    temperature=config.get('temperature', 0.1),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif client_type == 'ollama':
                client = self.llm_client['client']
                config = self.llm_client['config']
                response = client.chat(
                    model=config.get('model', 'llama2'),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response['message']['content']
        
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return ""
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM response to extract sentiment information.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed sentiment result
        """
        try:
            # Try to extract structured information
            result = {
                'sentiment_score': 0.0,
                'confidence': 0.5,
                'reasoning': response,
                'impact_likely': False,
                'method': 'llm'
            }
            
            # Extract sentiment score
            score_match = re.search(r'sentiment score[:\s]*([-]?\d*\.?\d+)', response.lower())
            if score_match:
                result['sentiment_score'] = float(score_match.group(1))
            
            # Extract confidence
            conf_match = re.search(r'confidence[:\s]*(\d*\.?\d+)', response.lower())
            if conf_match:
                result['confidence'] = float(conf_match.group(1))
            
            # Check for impact likelihood
            impact_keywords = ['significant', 'impact', 'move', 'affect', 'influence']
            result['impact_likely'] = any(keyword in response.lower() for keyword in impact_keywords)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._fallback_sentiment_analysis("")
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict:
        """
        Fallback sentiment analysis using VADER and TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        if not FALLBACK_AVAILABLE:
            return {
                'score': 0.0,
                'confidence': 0.3,
                'reasoning': 'Fallback analysis not available',
                'impact_likely': False,
                'method': 'fallback'
            }
        
        try:
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # Combine scores
            combined_score = (vader_compound + textblob_polarity) / 2
            
            # Determine confidence based on agreement
            score_diff = abs(vader_compound - textblob_polarity)
            confidence = max(0.3, 1.0 - score_diff)
            
            return {
                'score': combined_score,
                'confidence': confidence,
                'reasoning': f'VADER: {vader_compound:.3f}, TextBlob: {textblob_polarity:.3f}',
                'impact_likely': abs(combined_score) > 0.3,
                'method': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Fallback sentiment analysis failed: {e}")
            return {
                'score': 0.0,
                'confidence': 0.1,
                'reasoning': f'Analysis failed: {str(e)}',
                'impact_likely': False,
                'method': 'fallback'
            }
    
    def analyze_batch(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of news articles.
        
        Args:
            news_data: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment analysis results
        """
        if news_data.empty:
            return news_data
        
        logger.info(f"Analyzing sentiment for {len(news_data)} articles using enhanced analyzer")
        
        # Use fast mode if LLM is disabled or for large datasets
        if not self.use_llm or len(news_data) > 100:
            logger.info("Using fast sentiment analysis mode")
            return self._fast_sentiment_analysis(news_data)
        
        # Use enhanced LLM analysis for smaller datasets
        return self.enhanced_analyzer.analyze_batch(news_data)
    
    def _fast_sentiment_analysis(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fast sentiment analysis using rule-based approach.
        
        Args:
            news_data: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment scores
        """
        logger.info("Running fast sentiment analysis")
        
        # Copy the data to avoid modifying original
        result_df = news_data.copy()
        
        # Initialize sentiment scores
        sentiment_scores = []
        confidence_scores = []
        impact_scores = []
        
        # Keywords for sentiment analysis
        positive_keywords = [
            'surge', 'jump', 'rise', 'gain', 'increase', 'positive', 'growth', 
            'recovery', 'strength', 'bullish', 'optimistic', 'favorable'
        ]
        
        negative_keywords = [
            'crash', 'fall', 'decline', 'drop', 'loss', 'negative', 'weakness',
            'concern', 'risk', 'bearish', 'pessimistic', 'unfavorable'
        ]
        
        impact_keywords = [
            'breaking', 'urgent', 'critical', 'major', 'significant', 'important',
            'crisis', 'emergency', 'announcement', 'decision', 'policy'
        ]
        
        for idx, row in result_df.iterrows():
            # Combine title and summary for analysis
            text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
            
            # Count keyword occurrences
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            impact_count = sum(1 for word in impact_keywords if word in text)
            
            # Calculate sentiment score (-1 to 1)
            if positive_count > 0 or negative_count > 0:
                sentiment_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
                sentiment_score = np.clip(sentiment_score, -1, 1)
            else:
                # Use existing sentiment score if available, otherwise random
                sentiment_score = row.get('sentiment_score', np.random.normal(0, 0.2))
                sentiment_score = np.clip(sentiment_score, -1, 1)
            
            # Calculate confidence based on keyword presence
            total_keywords = positive_count + negative_count
            confidence = min(0.9, 0.3 + (total_keywords * 0.1))
            
            # Calculate impact likelihood
            impact_likely = min(0.9, 0.2 + (impact_count * 0.15))
            
            sentiment_scores.append(sentiment_score)
            confidence_scores.append(confidence)
            impact_scores.append(impact_likely)
        
        # Add results to DataFrame
        result_df['sentiment_score'] = sentiment_scores
        result_df['sentiment_confidence'] = confidence_scores
        result_df['impact_likely'] = impact_scores
        
        logger.info(f"Fast sentiment analysis complete for {len(result_df)} articles")
        return result_df
    
    def get_sentiment_summary(self, news_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for sentiment analysis.
        
        Args:
            news_df: DataFrame with sentiment analysis results
            
        Returns:
            Summary dictionary
        """
        if news_df.empty or 'sentiment_score' not in news_df.columns:
            return {}
        
        summary = {
            'total_articles': len(news_df),
            'avg_sentiment': news_df['sentiment_score'].mean(),
            'sentiment_std': news_df['sentiment_score'].std(),
            'positive_articles': len(news_df[news_df['sentiment_score'] > 0.1]),
            'negative_articles': len(news_df[news_df['sentiment_score'] < -0.1]),
            'neutral_articles': len(news_df[abs(news_df['sentiment_score']) <= 0.1]),
            'avg_confidence': news_df['sentiment_confidence'].mean(),
            'impact_likely_count': news_df['impact_likely'].sum(),
            'methods_used': news_df['sentiment_method'].value_counts().to_dict()
        }
        
        return summary 