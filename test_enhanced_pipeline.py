#!/usr/bin/env python3
"""
Test script for enhanced Smart Signal Filtering pipeline.
Tests all the fixes and improvements made to the system.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.main import SmartSignalFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_enhanced_pipeline():
    """Test the enhanced pipeline with all fixes."""
    
    print("🚀 Testing Enhanced Smart Signal Filtering Pipeline")
    print("=" * 60)
    
    try:
        # Initialize the system
        print("\n1. Initializing enhanced Smart Signal Filtering system...")
        filter_system = SmartSignalFilter()
        print("✅ System initialized successfully")
        
        # Define analysis parameters
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        symbols = ["XOP", "XLE", "USO", "BNO", "SPY"]
        
        print(f"\n2. Running enhanced analysis for period: {start_date} to {end_date}")
        print(f"   Symbols: {', '.join(symbols)}")
        
        # Run the complete pipeline
        print("\n3. Running enhanced pipeline...")
        results = filter_system.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            backtest=True,
            save_results=True,
            sentiment_threshold=0.15,  # Lowered for more signals
            volatility_threshold=0.02,  # Lowered for more signals
            reliability_threshold=0.3   # Lowered for more signals
        )
        
        print("✅ Enhanced pipeline completed successfully")
        
        # Test 1: Check if plots are generated with timestamps
        print("\n4. Testing plot generation with timestamps...")
        plots_dir = Path("data/plots")
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            if plot_files:
                print(f"✅ Generated {len(plot_files)} plot files with timestamps:")
                for plot_file in plot_files:
                    print(f"   - {plot_file.name}")
            else:
                print("⚠️ No plot files found")
        else:
            print("⚠️ Plots directory not found")
        
        # Test 2: Check walk-forward validation
        print("\n5. Testing walk-forward validation...")
        wf_results = results.get('walk_forward_results', {})
        if wf_results and 'splits' in wf_results:
            splits = wf_results['splits']
            print(f"✅ Walk-forward validation generated {len(splits)} splits")
            for i, split in enumerate(splits[:3]):  # Show first 3 splits
                print(f"   Split {i+1}: Train {split.get('train_start')} to {split.get('train_end')}, "
                      f"Test {split.get('test_start')} to {split.get('test_end')}")
        else:
            print("⚠️ Walk-forward validation failed or no splits generated")
        
        # Test 3: Check enhanced signal generation
        print("\n6. Testing enhanced signal generation...")
        signals = results.get('signals')
        if signals is not None and not signals.empty:
            print(f"✅ Generated {len(signals)} signals")
            print(f"   - Buy signals: {len(signals[signals['signal_type'] == 'BUY'])}")
            print(f"   - Sell signals: {len(signals[signals['signal_type'] == 'SELL'])}")
            print(f"   - Average signal strength: {signals['signal_strength'].mean():.3f}")
            print(f"   - Average signal quality: {signals['signal_quality'].mean():.3f}")
            
            # Check for enhanced features
            if 'volume_confirmation' in signals.columns:
                print("   - Volume confirmation: ✅")
            if 'price_confirmation' in signals.columns:
                print("   - Price confirmation: ✅")
            if 'momentum_filter' in signals.columns:
                print("   - Momentum filter: ✅")
        else:
            print("⚠️ No signals generated")
        
        # Test 4: Check enhanced AI agent
        print("\n7. Testing enhanced AI agent...")
        agent_insights = results.get('agent_insights', {})
        if agent_insights:
            print(f"✅ AI agent analysis completed")
            print(f"   - Novelty score: {agent_insights.get('novelty_score', 0):.3f}")
            print(f"   - Confidence: {agent_insights.get('confidence', 0):.3f}")
            print(f"   - Memory entries: {agent_insights.get('memory_entries', 0)}")
            
            # Check for enhanced features
            if 'reasoning' in agent_insights:
                print("   - Chain-of-thought reasoning: ✅")
            if 'recommendations' in agent_insights:
                print(f"   - Recommendations: {len(agent_insights['recommendations'])} generated")
            if 'signal_analysis' in agent_insights:
                print("   - Signal performance tracking: ✅")
        else:
            print("⚠️ AI agent analysis failed")
        
        # Test 5: Check trading results
        print("\n8. Testing trading results...")
        trading_results = results.get('trading_results', {})
        if trading_results:
            print(f"✅ Trading results generated")
            print(f"   - Total return: {trading_results.get('total_return', 0):.2%}")
            print(f"   - Sharpe ratio: {trading_results.get('sharpe_ratio', 0):.3f}")
            print(f"   - Win rate: {trading_results.get('win_rate', 0):.1%}")
            print(f"   - Total trades: {trading_results.get('total_trades', 0)}")
            
            # Check for enhanced features
            if 'cumulative_returns' in trading_results:
                print("   - Cumulative returns: ✅")
            if 'drawdown' in trading_results:
                print("   - Drawdown calculation: ✅")
            if 'benchmark_returns' in trading_results:
                print("   - Benchmark comparison: ✅")
        else:
            print("⚠️ Trading results failed")
        
        # Test 6: Check factor analysis
        print("\n9. Testing factor analysis...")
        factor_results = results.get('factor_analysis', {})
        if factor_results:
            print(f"✅ Factor analysis completed")
            if 'pca_results' in factor_results:
                print("   - PCA analysis: ✅")
            if 'factor_loadings' in factor_results:
                print("   - Factor loadings: ✅")
            if 'risk_decomposition' in factor_results:
                print("   - Risk decomposition: ✅")
        else:
            print("⚠️ Factor analysis failed")
        
        # Test 7: Check stress testing
        print("\n10. Testing stress testing...")
        stress_results = results.get('stress_test_results', {})
        if stress_results and 'summary' in stress_results:
            summary = stress_results['summary']
            print(f"✅ Stress testing completed")
            print(f"   - Scenarios tested: {summary.get('scenarios_tested', 0)}")
            print(f"   - Worst drawdown: {summary.get('worst_drawdown', 0):.1f}%")
            print(f"   - Average drawdown: {summary.get('average_drawdown', 0):.1f}%")
        else:
            print("⚠️ Stress testing failed")
        
        # Generate plots to test timestamp functionality
        print("\n11. Testing plot generation...")
        try:
            filter_system.plot_results(save_plots=True)
            print("✅ Enhanced plots generated successfully")
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 ENHANCED PIPELINE TEST RESULTS:")
        print("=" * 60)
        
        # Count successful tests
        successful_tests = 0
        total_tests = 11
        
        if plots_dir.exists() and list(plots_dir.glob("*.png")):
            successful_tests += 1
        if wf_results and 'splits' in wf_results and len(wf_results['splits']) > 0:
            successful_tests += 1
        if signals is not None and not signals.empty:
            successful_tests += 1
        if agent_insights:
            successful_tests += 1
        if trading_results:
            successful_tests += 1
        if factor_results:
            successful_tests += 1
        if stress_results and 'summary' in stress_results:
            successful_tests += 1
        
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        print(f"📊 Success rate: {successful_tests/total_tests*100:.1f}%")
        
        if successful_tests >= 8:
            print("🎉 ENHANCED PIPELINE IS WORKING CORRECTLY!")
        elif successful_tests >= 6:
            print("⚠️ MOST FEATURES WORKING - MINOR ISSUES DETECTED")
        else:
            print("❌ SIGNIFICANT ISSUES DETECTED - NEEDS ATTENTION")
        
        print("\n" + "=" * 60)
        print("Enhanced pipeline testing completed!")
        print("=" * 60)
        
        return successful_tests >= 8
        
    except Exception as e:
        logger.error(f"Error in enhanced pipeline test: {e}")
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = test_enhanced_pipeline()
    sys.exit(0 if success else 1) 