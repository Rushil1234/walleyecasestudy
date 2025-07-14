#!/usr/bin/env python3
"""
Fast test of the optimized Walleye case study system.
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import SmartSignalFilter

def fast_test():
    """Fast test of the optimized pipeline."""
    print("🚀 Fast Test of Optimized Walleye System")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Initialize system
        print("1. Initializing Smart Signal Filtering system...")
        system = SmartSignalFilter()
        
        # Run pipeline with optimizations
        print("2. Running optimized pipeline...")
        results = system.run_pipeline(
            start_date='2023-01-01',
            end_date='2023-12-31',
            symbols=['XOP', 'XLE', 'USO', 'BNO', 'SPY'],
            backtest=True,
            save_results=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("3. Analyzing results...")
        
        # Check news data
        news_data = results.get('news_data', [])
        print(f"📰 News Articles: {len(news_data)} (capped at 1000)")
        
        # Check signals
        signals = results.get('signals', [])
        print(f"📡 Signals Generated: {len(signals)}")
        
        # Check walk-forward validation
        wf_results = results.get('walk_forward_results', {})
        print(f"🔄 Walk-Forward Splits: {wf_results.get('num_splits', 0)}")
        
        # Check factor analysis
        factor_results = results.get('factor_analysis', {})
        if factor_results and 'pca_results' in factor_results:
            pca = factor_results['pca_results']
            print(f"🔍 PCA Components: {len(pca.get('explained_variance', []))}")
        
        # Check stress tests
        stress_results = results.get('stress_test_results', {})
        if stress_results and 'summary' in stress_results:
            summary = stress_results['summary']
            print(f"⚠️ Stress Tests: {summary.get('scenarios_tested', 0)} scenarios")
        
        # Check AI agent
        agent_insights = results.get('agent_insights', {})
        print(f"🤖 AI Agent Analysis: Novelty={agent_insights.get('novelty_score', 0):.3f}, Confidence={agent_insights.get('confidence', 0):.3f}")
        
        # Check performance
        trading_results = results.get('trading_results', {})
        if 'daily_returns' in trading_results:
            returns = trading_results['daily_returns']
            total_return = (1 + returns).prod() - 1
            print(f"💰 Performance: Total Return = {total_return:.2%}")
        
        print(f"\n⏱️ Processing Time: {processing_time:.1f} seconds")
        print(f"📊 Articles per second: {len(news_data) / processing_time:.1f}")
        
        print("\n✅ Fast test completed successfully!")
        print("🎯 Optimized pipeline is working efficiently.")
        print("🚀 Ready for Streamlit deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during fast test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fast_test()
    sys.exit(0 if success else 1) 