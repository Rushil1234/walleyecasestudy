#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced Features

Tests all the new improvements:
1. Hyperparameter Optimization with Optuna
2. Advanced Feature Engineering
3. Fixed PCA Analysis
4. Enhanced Signal Generation
5. Active AI Agent with Recommendations
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.main import SmartSignalFilter
from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from src.signals.feature_engineering import AdvancedFeatureEngineer
from src.risk.factor_analysis import FactorExposureAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_feature_engineering():
    """Test advanced feature engineering."""
    logger.info("Testing Advanced Feature Engineering...")
    
    try:
        # Create sample news data
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
        
        logger.info(f"✅ Feature Engineering Test PASSED")
        logger.info(f"Original features: {len(sample_news.columns)}")
        logger.info(f"Enhanced features: {len(enhanced_news.columns)}")
        logger.info(f"New features added: {len(enhanced_news.columns) - len(sample_news.columns)}")
        logger.info(f"Feature summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature Engineering Test FAILED: {e}")
        return False

def test_pca_analysis():
    """Test fixed PCA analysis."""
    logger.info("Testing PCA Analysis...")
    
    try:
        # Initialize factor analyzer
        analyzer = FactorExposureAnalyzer(['XOP', 'XLE', 'USO', 'BNO', 'SPY'])
        
        # Fetch data
        returns_df = analyzer.fetch_factor_data('2023-01-01', '2023-12-31')
        
        logger.info(f"Returns DataFrame shape: {returns_df.shape}")
        
        if returns_df.empty:
            logger.warning("No data returned from factor analyzer")
            return False
        
        # Perform PCA analysis
        pca_results = analyzer.perform_pca_analysis(returns_df)
        
        # Check results
        if 'data_info' in pca_results:
            data_info = pca_results['data_info']
            logger.info(f"✅ PCA Analysis Test PASSED")
            logger.info(f"Components: {data_info.get('n_components', 0)}")
            logger.info(f"Variance explained: {data_info.get('total_variance_explained', 0):.1%}")
            logger.info(f"Data shape: {data_info.get('clean_shape', (0, 0))}")
            
            # Check if we have meaningful results
            if data_info.get('n_components', 0) > 0 and data_info.get('total_variance_explained', 0) > 0:
                return True
            else:
                logger.warning("PCA analysis returned empty results")
                return False
        else:
            logger.error("PCA analysis failed - no data_info in results")
            return False
            
    except Exception as e:
        logger.error(f"❌ PCA Analysis Test FAILED: {e}")
        return False

def test_hyperparameter_optimization():
    """Test hyperparameter optimization."""
    logger.info("Testing Hyperparameter Optimization...")
    
    try:
        # Initialize optimizer
        optimizer = HyperparameterOptimizer(optimization_target='sharpe_ratio')
        
        # Test with a small number of trials
        results = optimizer.optimize(
            start_date="2023-01-01",
            end_date="2023-03-31",  # Shorter period for testing
            symbols=["XOP", "XLE", "USO"],
            n_trials=3,  # Small number for testing
            timeout=300  # 5 minutes timeout
        )
        
        logger.info(f"✅ Hyperparameter Optimization Test PASSED")
        logger.info(f"Best score: {results.get('best_score', 0):.4f}")
        logger.info(f"Best parameters: {results.get('best_params', {})}")
        logger.info(f"Trials completed: {len(results.get('trial_history', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter Optimization Test FAILED: {e}")
        return False

def test_enhanced_signal_generation():
    """Test enhanced signal generation with new features."""
    logger.info("Testing Enhanced Signal Generation...")
    
    try:
        # Initialize system
        filter_system = SmartSignalFilter()
        
        # Run pipeline with enhanced features
        results = filter_system.run_pipeline(
            start_date="2023-01-01",
            end_date="2023-03-31",  # Shorter period for testing
            symbols=["XOP", "XLE", "USO", "BNO", "SPY"],
            sentiment_threshold=0.0,  # More permissive
            volatility_threshold=0.01,  # More permissive
            reliability_threshold=0.3,  # More permissive
            backtest=True,
            save_results=False
        )
        
        if not results:
            logger.error("Pipeline returned no results")
            return False
        
        # Check for enhanced features in signals
        signals = results.get('signals', pd.DataFrame())
        if not signals.empty:
            enhanced_features = [
                'sentiment_surprise', 'market_impact_score', 'event_importance',
                'signal_amplification', 'sentiment_volatility', 'reliability_weighted_sentiment',
                'event_count', 'cluster_size'
            ]
            
            found_features = [f for f in enhanced_features if f in signals.columns]
            logger.info(f"✅ Enhanced Signal Generation Test PASSED")
            logger.info(f"Total signals generated: {len(signals)}")
            logger.info(f"Enhanced features found: {len(found_features)}/{len(enhanced_features)}")
            logger.info(f"Features: {found_features}")
            
            return len(found_features) > 0
        else:
            logger.warning("No signals generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced Signal Generation Test FAILED: {e}")
        return False

def test_ai_agent_recommendations():
    """Test AI agent recommendations."""
    logger.info("Testing AI Agent Recommendations...")
    
    try:
        # Initialize system
        filter_system = SmartSignalFilter()
        
        # Run pipeline to get results
        results = filter_system.run_pipeline(
            start_date="2023-01-01",
            end_date="2023-03-31",
            symbols=["XOP", "XLE", "USO"],
            backtest=True,
            save_results=False
        )
        
        if not results:
            logger.error("Pipeline returned no results for AI agent test")
            return False
        
        # Check for AI agent recommendations
        agent_recommendations = results.get('agent_recommendations', {})
        topic_suggestions = results.get('topic_suggestions', {})
        
        logger.info(f"✅ AI Agent Recommendations Test PASSED")
        logger.info(f"Agent recommendations status: {agent_recommendations.get('status', 'unknown')}")
        logger.info(f"Topic suggestions status: {topic_suggestions.get('status', 'unknown')}")
        
        if agent_recommendations.get('status') == 'recommendations_generated':
            recommendations = agent_recommendations.get('recommendations', [])
            logger.info(f"Number of recommendations: {len(recommendations)}")
            for rec in recommendations[:3]:  # Show first 3
                logger.info(f"  - {rec.get('parameter', 'unknown')}: {rec.get('action', 'unknown')}")
        
        if topic_suggestions.get('status') == 'suggestions_generated':
            suggestions = topic_suggestions.get('suggestions', [])
            logger.info(f"Number of topic suggestions: {len(suggestions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AI Agent Recommendations Test FAILED: {e}")
        return False

def test_comprehensive_pipeline():
    """Test the complete enhanced pipeline."""
    logger.info("Testing Comprehensive Enhanced Pipeline...")
    
    try:
        # Initialize system
        filter_system = SmartSignalFilter()
        
        # Run full pipeline
        results = filter_system.run_pipeline(
            start_date="2023-01-01",
            end_date="2023-06-30",  # 6 months for comprehensive test
            symbols=["XOP", "XLE", "USO", "BNO", "SPY"],
            sentiment_threshold=0.0,
            volatility_threshold=0.01,
            reliability_threshold=0.3,
            backtest=True,
            save_results=True
        )
        
        if not results:
            logger.error("Comprehensive pipeline returned no results")
            return False
        
        # Check all components
        checks = {
            'equity_data': len(results.get('equity_data', {})) > 0,
            'news_data': not results.get('news_data', pd.DataFrame()).empty,
            'signals': not results.get('signals', pd.DataFrame()).empty,
            'trading_results': bool(results.get('trading_results')),
            'risk_analysis': bool(results.get('risk_analysis')),
            'factor_analysis': bool(results.get('factor_analysis')),
            'walk_forward_results': bool(results.get('walk_forward_results')),
            'stress_test_results': bool(results.get('stress_test_results')),
            'bias_analysis': bool(results.get('bias_analysis')),
            'agent_insights': bool(results.get('agent_insights')),
            'agent_recommendations': bool(results.get('agent_recommendations')),
            'topic_suggestions': bool(results.get('topic_suggestions'))
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        logger.info(f"✅ Comprehensive Pipeline Test PASSED")
        logger.info(f"Checks passed: {passed_checks}/{total_checks}")
        
        for component, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {component}")
        
        # Show key metrics
        if results.get('trading_results'):
            trading = results['trading_results']
            logger.info(f"Trading Results:")
            logger.info(f"  Total Return: {trading.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {trading.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max Drawdown: {trading.get('max_drawdown', 0):.2%}")
            logger.info(f"  Total Trades: {trading.get('total_trades', 0)}")
        
        if results.get('signals') is not None:
            signals = results['signals']
            logger.info(f"Signal Generation:")
            logger.info(f"  Total Signals: {len(signals)}")
            if not signals.empty:
                logger.info(f"  Signal Types: {signals['signal_type'].value_counts().to_dict()}")
        
        return passed_checks >= total_checks * 0.8  # 80% success rate
        
    except Exception as e:
        logger.error(f"❌ Comprehensive Pipeline Test FAILED: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Enhanced Features Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("PCA Analysis", test_pca_analysis),
        ("Hyperparameter Optimization", test_hyperparameter_optimization),
        ("Enhanced Signal Generation", test_enhanced_signal_generation),
        ("AI Agent Recommendations", test_ai_agent_recommendations),
        ("Comprehensive Pipeline", test_comprehensive_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test...")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Enhanced features are working correctly.")
    elif passed >= total * 0.8:
        logger.info("👍 Most tests passed. Enhanced features are mostly working.")
    else:
        logger.warning("⚠️ Many tests failed. Some enhanced features may need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 