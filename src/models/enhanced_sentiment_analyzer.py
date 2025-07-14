"""
Enhanced Sentiment Analyzer with Real LLM Integration and Advanced NLP

Uses Ollama + Mistral for local LLM sentiment analysis and advanced NLP features
for novelty detection, entity extraction, and semantic analysis.
"""

import ollama
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from datetime import datetime
import json
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)


class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analyzer with real LLM integration and advanced NLP features.
    """
    
    def __init__(self, model_name: str = "mistral", use_llm: bool = True):
        """
        Initialize the enhanced sentiment analyzer.
        
        Args:
            model_name: Ollama model name (default: mistral)
            use_llm: Whether to use LLM for sentiment analysis
        """
        self.model_name = model_name
        self.use_llm = use_llm
        
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Handle NLTK data loading with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not available, using basic stopwords")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Oil and geopolitical keywords for specialized analysis
        self.oil_keywords = {
            'oil', 'crude', 'petroleum', 'OPEC', 'OPEC+', 'WTI', 'Brent', 
            'futures', 'drilling', 'fracking', 'shale', 'pipeline', 'refinery',
            'tanker', 'production', 'supply', 'demand', 'inventory'
        }
        
        self.geopolitical_keywords = {
            'Iran', 'Saudi Arabia', 'Russia', 'Venezuela', 'Middle East',
            'Persian Gulf', 'Strait of Hormuz', 'sanctions', 'embargo',
            'conflict', 'war', 'attack', 'strike', 'tension', 'diplomacy'
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
            logger.info(f"✅ LLM connection successful with {self.model_name}")
        except Exception as e:
            logger.error(f"❌ LLM connection failed: {e}")
            self.use_llm = False
            logger.info("Falling back to rule-based sentiment analysis")
    
    def analyze_sentiment_llm(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using Ollama LLM with chain-of-thought reasoning.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.use_llm:
            return self._fallback_sentiment_analysis(text)
        
        try:
            # Create detailed prompt for sentiment analysis
            prompt = f"""
            Analyze the sentiment of the following news headline/text related to oil markets and geopolitics.
            
            Text: "{text}"
            
            Please provide a detailed analysis including:
            1. Overall sentiment (positive/negative/neutral)
            2. Sentiment score (-1 to +1, where -1 is very negative, +1 is very positive)
            3. Confidence level (0 to 1)
            4. Key entities mentioned (countries, companies, people)
            5. Potential market impact (high/medium/low)
            6. Reasoning for your assessment
            
            Format your response as JSON:
            {{
                "sentiment": "positive/negative/neutral",
                "score": -1.0 to 1.0,
                "confidence": 0.0 to 1.0,
                "entities": ["entity1", "entity2"],
                "market_impact": "high/medium/low",
                "reasoning": "detailed explanation"
            }}
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            # Parse JSON response
            try:
                result = json.loads(response['message']['content'])
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("LLM response not in JSON format, using fallback")
                return self._fallback_sentiment_analysis(text)
                
        except Exception as e:
            logger.error(f"Error in LLM sentiment analysis: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback sentiment analysis using rule-based methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Combine scores
        combined_score = (vader_scores['compound'] + textblob_polarity) / 2
        
        # Determine sentiment
        if combined_score > 0.1:
            sentiment = "positive"
        elif combined_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Extract entities using spaCy
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'ORG', 'PERSON']]
        
        # Assess market impact based on keywords
        market_impact = self._assess_market_impact(text)
        
        return {
            "sentiment": sentiment,
            "score": combined_score,
            "confidence": 1 - textblob_subjectivity,  # Lower subjectivity = higher confidence
            "entities": entities,
            "market_impact": market_impact,
            "reasoning": f"Rule-based analysis: VADER compound={vader_scores['compound']:.3f}, TextBlob polarity={textblob_polarity:.3f}",
            "vader_scores": vader_scores,
            "textblob_scores": {"polarity": textblob_polarity, "subjectivity": textblob_subjectivity}
        }
    
    def _assess_market_impact(self, text: str) -> str:
        """
        Assess potential market impact based on keywords and context.
        
        Args:
            text: Text to analyze
            
        Returns:
            Market impact level (high/medium/low)
        """
        text_lower = text.lower()
        
        # High impact keywords
        high_impact = {
            'war', 'attack', 'sanctions', 'embargo', 'production cut', 'supply disruption',
            'pipeline attack', 'refinery fire', 'tanker seizure', 'OPEC decision'
        }
        
        # Medium impact keywords
        medium_impact = {
            'tension', 'diplomatic', 'negotiation', 'meeting', 'talks', 'agreement',
            'production increase', 'inventory', 'demand forecast'
        }
        
        # Count high and medium impact keywords
        high_count = sum(1 for keyword in high_impact if keyword in text_lower)
        medium_count = sum(1 for keyword in medium_impact if keyword in text_lower)
        
        if high_count > 0:
            return "high"
        elif medium_count > 0:
            return "medium"
        else:
            return "low"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with entity types and lists of entities
        """
        doc = self.nlp(text)
        
        entities = {
            'countries': [],
            'organizations': [],
            'people': [],
            'locations': [],
            'dates': [],
            'money': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'GPE':  # Countries, cities, states
                entities['countries'].append(ent.text)
            elif ent.label_ == 'ORG':  # Organizations
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'PERSON':  # People
                entities['people'].append(ent.text)
            elif ent.label_ == 'LOC':  # Locations
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':  # Dates
                entities['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':  # Money amounts
                entities['money'].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def calculate_novelty_score(self, text: str, historical_texts: List[str]) -> float:
        """
        Calculate novelty score based on semantic similarity to historical texts.
        
        Args:
            text: Current text to analyze
            historical_texts: List of historical texts for comparison
            
        Returns:
            Novelty score (0-1, where 1 is most novel)
        """
        if not historical_texts:
            return 0.5  # Default score if no historical data
        
        # Preprocess current text
        current_tokens = self._preprocess_text(text)
        
        # Calculate similarity scores
        similarities = []
        for hist_text in historical_texts[-50:]:  # Compare with last 50 texts
            hist_tokens = self._preprocess_text(hist_text)
            similarity = self._calculate_jaccard_similarity(current_tokens, hist_tokens)
            similarities.append(similarity)
        
        # Novelty is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0
        novelty_score = 1 - avg_similarity
        
        return novelty_score
    
    def _preprocess_text(self, text: str) -> set:
        """
        Preprocess text for similarity calculation.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Set of preprocessed tokens
        """
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalnum() and token not in self.stop_words
        ]
        
        return set(tokens)
    
    def _calculate_jaccard_similarity(self, set1: set, set2: set) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity score (0-1)
        """
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from text using spaCy.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key phrases
        """
        doc = self.nlp(text)
        
        # Extract noun chunks and named entities
        key_phrases = []
        
        # Noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases only
                key_phrases.append(chunk.text)
        
        # Named entities
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'ORG', 'PERSON', 'LOC']:
                key_phrases.append(ent.text)
        
        # Remove duplicates and sort by length
        key_phrases = list(set(key_phrases))
        key_phrases.sort(key=len, reverse=True)
        
        return key_phrases[:10]  # Return top 10 phrases
    
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
        
        logger.info(f"Analyzing sentiment for {len(news_data)} articles")
        
        results = []
        historical_texts = []
        
        for idx, row in news_data.iterrows():
            try:
                # Combine title and summary for analysis
                text = f"{row.get('title', '')} {row.get('summary', '')}"
                
                # Analyze sentiment
                sentiment_result = self.analyze_sentiment_llm(text)
                
                # Extract entities
                entities = self.extract_entities(text)
                
                # Calculate novelty score
                novelty_score = self.calculate_novelty_score(text, historical_texts)
                
                # Extract key phrases
                key_phrases = self.extract_key_phrases(text)
                
                # Store historical text for novelty calculation
                historical_texts.append(text)
                
                # Combine results
                result = {
                    'sentiment_score': sentiment_result['score'],
                    'sentiment_label': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence'],
                    'market_impact': sentiment_result['market_impact'],
                    'novelty_score': novelty_score,
                    'entities': json.dumps(entities),
                    'key_phrases': json.dumps(key_phrases),
                    'reasoning': sentiment_result.get('reasoning', ''),
                    'analysis_method': 'llm' if self.use_llm else 'rule_based'
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing article {idx}: {e}")
                # Add fallback result
                results.append({
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.5,
                    'market_impact': 'low',
                    'novelty_score': 0.5,
                    'entities': '{}',
                    'key_phrases': '[]',
                    'reasoning': f'Error in analysis: {str(e)}',
                    'analysis_method': 'error'
                })
        
        # Add sentiment analysis results to original DataFrame
        sentiment_df = pd.DataFrame(results, index=news_data.index)
        result_df = pd.concat([news_data, sentiment_df], axis=1)
        
        logger.info(f"Sentiment analysis complete. LLM used: {self.use_llm}")
        return result_df
    
    def get_analysis_summary(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for sentiment analysis.
        
        Args:
            news_data: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if news_data.empty:
            return {}
        
        summary = {
            'total_articles': len(news_data),
            'avg_sentiment_score': news_data['sentiment_score'].mean(),
            'sentiment_distribution': news_data['sentiment_label'].value_counts().to_dict(),
            'avg_confidence': news_data['confidence'].mean(),
            'avg_novelty_score': news_data['novelty_score'].mean(),
            'market_impact_distribution': news_data['market_impact'].value_counts().to_dict(),
            'llm_usage_rate': (news_data['analysis_method'] == 'llm').mean(),
            'high_impact_articles': len(news_data[news_data['market_impact'] == 'high']),
            'high_novelty_articles': len(news_data[news_data['novelty_score'] > 0.7])
        }
        
        return summary 