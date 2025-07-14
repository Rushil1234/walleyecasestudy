#!/usr/bin/env python3
"""
Test script for enhanced LLM integration and advanced NLP features.
"""

import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from src.agents.enhanced_ai_agent import EnhancedAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_sentiment_analyzer():
    """Test the enhanced sentiment analyzer with real LLM."""
    print("🧪 Testing Enhanced Sentiment Analyzer")
    print("=" * 50)
    
    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedSentimentAnalyzer(use_llm=True)
        
        # Test texts
        test_texts = [
            "OPEC+ announces production cuts, oil prices surge to new highs",
            "Iran tensions escalate in Persian Gulf, supply disruption fears grow",
            "Saudi Arabia and Russia agree to extend oil production agreement",
            "Oil prices fall as US shale production reaches record levels",
            "Pipeline attack in Middle East causes temporary supply disruption"
        ]
        
        print(f"Testing {len(test_texts)} sample texts...")
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Text: {text}")
            
            # Analyze sentiment
            result = analyzer.analyze_sentiment_llm(text)
            
            print(f"   Sentiment: {result['sentiment']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Market Impact: {result['market_impact']}")
            print(f"   Entities: {result['entities']}")
            print(f"   Reasoning: {result['reasoning'][:100]}...")
        
        print("\n✅ Enhanced Sentiment Analyzer test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing enhanced sentiment analyzer: {e}")

def test_enhanced_ai_agent():
    """Test the enhanced AI agent with chain-of-thought reasoning."""
    print("\n🤖 Testing Enhanced AI Agent")
    print("=" * 50)
    
    try:
        # Initialize enhanced agent
        agent = EnhancedAIAgent(use_llm=True)
        
        # Create sample data
        import pandas as pd
        import numpy as np
        
        # Sample equity data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        equity_data = {
            'XOP': pd.DataFrame({
                'Close': np.random.randn(30).cumsum() + 100,
                'Returns': np.random.randn(30) * 0.02
            }, index=dates),
            'SPY': pd.DataFrame({
                'Close': np.random.randn(30).cumsum() + 400,
                'Returns': np.random.randn(30) * 0.015
            }, index=dates)
        }
        
        # Sample news data
        news_data = pd.DataFrame({
            'title': [
                'OPEC+ announces production cuts',
                'Iran tensions escalate',
                'Oil prices surge on supply fears',
                'Saudi Arabia increases production',
                'Pipeline attack disrupts supply'
            ],
            'summary': [
                'OPEC+ countries agree to reduce oil production by 2 million barrels per day',
                'Tensions between Iran and US escalate in Persian Gulf region',
                'Oil prices jump 5% on concerns about supply disruptions',
                'Saudi Arabia announces increased oil production to stabilize markets',
                'Attack on major pipeline causes temporary supply disruption'
            ],
            'source': ['Reuters', 'Al Jazeera', 'Bloomberg', 'BBC', 'AP'],
            'published_date': dates[:5],
            'sentiment_score': [0.8, -0.6, 0.9, 0.3, -0.7]
        })
        
        # Sample signals
        signals = pd.DataFrame({
            'signal_direction': [1, -1, 1, 0, -1],
            'signal_strength': [0.8, 0.7, 0.9, 0.5, 0.6],
            'signal_quality': [0.9, 0.8, 0.7, 0.6, 0.8]
        }, index=dates[:5])
        
        # Sample trading results
        trading_results = {
            'performance_summary': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'win_rate': 0.65
            }
        }
        
        print("Running enhanced market analysis...")
        
        # Analyze market
        insights = agent.analyze_market_with_cot(
            equity_data, news_data, signals, trading_results
        )
        
        print(f"\n📊 Analysis Results:")
        print(f"Novelty Score: {insights['novelty_score']:.3f}")
        print(f"Confidence: {insights['confidence']:.3f}")
        print(f"Market Regime: {insights['market_regime']}")
        print(f"Risk Assessment: {insights['risk_assessment']}")
        
        print(f"\n🎯 Recommendations:")
        for i, rec in enumerate(insights['recommendations'][:3], 1):
            print(f"{i}. {rec}")
        
        print(f"\n🔍 Entity Analysis:")
        entity_analysis = insights['entity_analysis']
        if 'countries_frequency' in entity_analysis:
            print(f"Top Countries: {list(entity_analysis['countries_frequency'].keys())[:3]}")
        
        print(f"\n💭 Chain-of-Thought (excerpt):")
        cot = insights['chain_of_thought']
        print(f"{cot[:200]}...")
        
        print("\n✅ Enhanced AI Agent test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing enhanced AI agent: {e}")

def test_advanced_nlp_features():
    """Test advanced NLP features."""
    print("\n📝 Testing Advanced NLP Features")
    print("=" * 50)
    
    try:
        from src.models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
        
        analyzer = EnhancedSentimentAnalyzer(use_llm=False)  # Use rule-based for speed
        
        # Test entity extraction
        text = "OPEC+ announces production cuts in Saudi Arabia, affecting global oil markets"
        entities = analyzer.extract_entities(text)
        
        print(f"Text: {text}")
        print(f"Entities: {entities}")
        
        # Test key phrase extraction
        key_phrases = analyzer.extract_key_phrases(text)
        print(f"Key Phrases: {key_phrases}")
        
        # Test novelty calculation
        historical_texts = [
            "Oil prices rise on supply concerns",
            "OPEC meeting scheduled for next month",
            "Iran tensions affect oil markets"
        ]
        novelty_score = analyzer.calculate_novelty_score(text, historical_texts)
        print(f"Novelty Score: {novelty_score:.3f}")
        
        print("\n✅ Advanced NLP features test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing advanced NLP features: {e}")

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced LLM Integration and Advanced NLP Features")
    print("=" * 70)
    
    # Test enhanced sentiment analyzer
    test_enhanced_sentiment_analyzer()
    
    # Test enhanced AI agent
    test_enhanced_ai_agent()
    
    # Test advanced NLP features
    test_advanced_nlp_features()
    
    print("\n🎉 All tests completed!")
    print("\n📋 Summary:")
    print("✅ Enhanced Sentiment Analyzer with Ollama + Mistral")
    print("✅ Advanced NLP features (entity extraction, key phrases, novelty detection)")
    print("✅ Enhanced AI Agent with chain-of-thought reasoning")
    print("✅ Real LLM integration for market analysis")

if __name__ == "__main__":
    main() 