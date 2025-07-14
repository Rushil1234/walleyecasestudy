#!/usr/bin/env python3
"""
Test script for enhanced factor analysis with SHAP and enhanced stress testing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import our enhanced modules
from risk.enhanced_factor_analysis import EnhancedFactorAnalyzer
from risk.enhanced_stress_tests import EnhancedStressTestManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_factor_analysis():
    """Test enhanced factor analysis with SHAP."""
    logger.info("Testing enhanced factor analysis with SHAP...")
    
    # Initialize analyzer
    analyzer = EnhancedFactorAnalyzer()
    
    # Run analysis for recent period
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    
    try:
        results = analyzer.run_complete_analysis(
            start_date=start_date,
            end_date=end_date,
            target_symbol='XOP'
        )
        
        # Print results
        print("\n" + "="*60)
        print("ENHANCED FACTOR ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nTraining Results:")
        print(f"Cross-validation R²: {results['training_results']['cv_r2_mean']:.4f} ± {results['training_results']['cv_r2_std']:.4f}")
        
        print(f"\nTop 5 Most Important Features (SHAP):")
        shap_importance = results['shap_results']['feature_importance']
        top_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
        
        print(f"\nStrategy Insights:")
        insights = results['insights']
        for category, items in insights.items():
            if items:
                print(f"\n{category.replace('_', ' ').title()}:")
                for item in items:
                    if isinstance(item, dict):
                        print(f"  - {item.get('description', item.get('factor', 'N/A'))}")
                    else:
                        print(f"  - {item}")
        
        print(f"\nData Info:")
        data_info = results['data_info']
        for key, value in data_info.items():
            print(f"  {key}: {value}")
        
        print(f"\nPlots saved to:")
        for plot_name, plot_path in results['plots'].items():
            print(f"  {plot_name}: {plot_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced factor analysis failed: {e}")
        return None

def test_enhanced_stress_testing():
    """Test enhanced stress testing with realistic scenarios."""
    logger.info("Testing enhanced stress testing...")
    
    # Initialize stress test manager
    stress_manager = EnhancedStressTestManager()
    
    # Create mock returns data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    mock_returns = pd.DataFrame({
        'returns': np.random.normal(0.001, 0.02, len(dates)),
        'cumulative': np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
    }, index=dates)
    
    # Define custom events
    custom_events = [
        {"label": "COVID-19 Crash", "start": "2020-02-15", "end": "2020-04-30"},
        {"label": "Russia-Ukraine War", "start": "2022-02-20", "end": "2022-04-15"},
        {"label": "Hypothetical Oil Collapse", "shock_pct": -0.30},
        {"label": "VIX Spike Scenario", "vix_change": 15.0},
    ]
    
    try:
        results = stress_manager.perform_enhanced_stress_testing(
            returns_df=mock_returns,
            events=custom_events
        )
        
        # Print results
        print("\n" + "="*60)
        print("ENHANCED STRESS TESTING RESULTS")
        print("="*60)
        
        print(f"\nScenarios Tested: {results['scenarios_tested']}")
        print(f"Data Period: {results['data_period']}")
        
        print(f"\nSummary Statistics:")
        summary = results['summary_statistics']
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nHistorical Scenarios:")
        for scenario_name, scenario_data in results['historical_scenarios'].items():
            print(f"\n  {scenario_name}:")
            for symbol, symbol_data in scenario_data.items():
                if 'metrics' in symbol_data:
                    metrics = symbol_data['metrics']
                    print(f"    {symbol}: Return={metrics['total_return']:.2%}, "
                          f"Drawdown={metrics['max_drawdown']:.2%}, "
                          f"Vol={metrics['volatility']:.2%}")
        
        print(f"\nHypothetical Scenarios:")
        for scenario_name, scenario_data in results['hypothetical_scenarios'].items():
            print(f"\n  {scenario_name}:")
            for symbol, symbol_data in scenario_data.items():
                if 'metrics' in symbol_data:
                    metrics = symbol_data['metrics']
                    print(f"    {symbol}: Return={metrics['total_return']:.2%}, "
                          f"Drawdown={metrics['max_drawdown']:.2%}, "
                          f"Vol={metrics['volatility']:.2%}")
        
        print(f"\nRolling Analysis:")
        rolling = results['rolling_analysis']
        for symbol, rolling_data in rolling.items():
            if isinstance(rolling_data, pd.DataFrame) and len(rolling_data) > 0:
                avg_drawdown = rolling_data['max_drawdown'].mean()
                avg_vol = rolling_data['volatility'].mean()
                print(f"  {symbol}: Avg Drawdown={avg_drawdown:.2%}, Avg Vol={avg_vol:.2%}")
        
        print(f"\nPlots saved to:")
        for plot_name, plot_path in results['plots'].items():
            print(f"  {plot_name}: {plot_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced stress testing failed: {e}")
        return None

def test_shap_integration():
    """Test SHAP integration with strategy improvement insights."""
    logger.info("Testing SHAP integration for strategy improvement...")
    
    # Initialize analyzer
    analyzer = EnhancedFactorAnalyzer()
    
    # Fetch data for a shorter period for faster testing
    start_date = '2023-06-01'
    end_date = '2024-12-31'
    
    try:
        # Fetch and prepare data
        data = analyzer.fetch_factor_data(start_date, end_date)
        features, target = analyzer.prepare_features(data, 'XOP')
        
        # Train model
        training_results = analyzer.train_gradient_boosted_model(features, target)
        
        # SHAP analysis
        shap_results = analyzer.analyze_feature_importance(features, target)
        
        # Generate strategy insights
        insights = analyzer.generate_strategy_insights(shap_results)
        
        print("\n" + "="*60)
        print("SHAP STRATEGY IMPROVEMENT INSIGHTS")
        print("="*60)
        
        print(f"\nModel Performance:")
        print(f"Cross-validation R²: {training_results['cv_r2_mean']:.4f} ± {training_results['cv_r2_std']:.4f}")
        
        print(f"\nTop Predictive Factors:")
        top_features = shap_results['top_features'][:5]
        for i, feature in enumerate(top_features, 1):
            importance = shap_results['feature_importance'][feature]
            print(f"  {i}. {feature} (SHAP importance: {importance:.4f})")
        
        print(f"\nStrategy Recommendations:")
        for recommendation in insights['recommendations']:
            print(f"  • {recommendation}")
        
        print(f"\nRisk Indicators:")
        for indicator in insights['risk_indicators']:
            print(f"  • {indicator['description']} (Importance: {indicator['importance']:.4f})")
        
        print(f"\nOpportunity Signals:")
        for signal in insights['opportunity_signals']:
            print(f"  • {signal['description']} (Importance: {signal['importance']:.4f})")
        
        return {
            'training_results': training_results,
            'shap_results': shap_results,
            'insights': insights
        }
        
    except Exception as e:
        logger.error(f"SHAP integration test failed: {e}")
        return None

def main():
    """Run all enhanced feature tests."""
    logger.info("Starting enhanced feature tests...")
    
    print("Testing Enhanced Factor Analysis with SHAP...")
    factor_results = test_enhanced_factor_analysis()
    
    print("\nTesting Enhanced Stress Testing...")
    stress_results = test_enhanced_stress_testing()
    
    print("\nTesting SHAP Integration for Strategy Improvement...")
    shap_results = test_shap_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if factor_results:
        print("✅ Enhanced Factor Analysis: SUCCESS")
        print(f"   - Model R²: {factor_results['training_results']['cv_r2_mean']:.4f}")
        print(f"   - Features analyzed: {factor_results['data_info']['n_features']}")
    else:
        print("❌ Enhanced Factor Analysis: FAILED")
    
    if stress_results:
        print("✅ Enhanced Stress Testing: SUCCESS")
        print(f"   - Scenarios tested: {stress_results['scenarios_tested']}")
        summary = stress_results.get('summary_statistics', {})
        if summary and 'worst_drawdown' in summary:
            print(f"   - Worst drawdown: {summary['worst_drawdown']:.2%}")
        else:
            print("   - Summary statistics: Not available")
    else:
        print("❌ Enhanced Stress Testing: FAILED")
    
    if shap_results:
        print("✅ SHAP Integration: SUCCESS")
        print(f"   - Model R²: {shap_results['training_results']['cv_r2_mean']:.4f}")
        print(f"   - Top factor: {shap_results['shap_results']['top_features'][0]}")
    else:
        print("❌ SHAP Integration: FAILED")
    
    print("\nEnhanced features test completed!")

if __name__ == "__main__":
    main() 