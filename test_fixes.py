#!/usr/bin/env python3
"""
Test script to verify signal generation fixes.
"""

import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.main import SmartSignalFilter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_signal_generation():
    """Test that signal generation now produces actual trades."""
    
    print("🧪 Testing Signal Generation Fixes")
    print("=" * 50)
    
    try:
        # Initialize the system
        filter_system = SmartSignalFilter()
        
        # Run a short test period
        start_date = "2023-06-01"
        end_date = "2023-08-31"
        symbols = ["XOP", "XLE", "USO", "BNO", "SPY"]
        
        print(f"Testing period: {start_date} to {end_date}")
        print(f"Symbols: {', '.join(symbols)}")
        
        # Run the pipeline
        results = filter_system.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            backtest=True,
            save_results=True
        )
        
        # Check signals
        signals = results.get('signals', pd.DataFrame())
        print(f"\n📊 Signal Generation Results:")
        print(f"   Total signals generated: {len(signals)}")
        
        if not signals.empty:
            print(f"   Signal date range: {signals.index.min()} to {signals.index.max()}")
            print(f"   Average signal strength: {signals['signal_strength'].mean():.3f}")
            print(f"   Average signal quality: {signals['signal_quality'].mean():.3f}")
            print(f"   Buy signals: {len(signals[signals['signal_direction'] > 0])}")
            print(f"   Sell signals: {len(signals[signals['signal_direction'] < 0])}")
        
        # Check trading results
        trading_results = results.get('trading_results', {})
        if trading_results:
            print(f"\n💰 Trading Results:")
            print(f"   Total trades: {trading_results.get('total_trades', 0)}")
            print(f"   Total return: {trading_results.get('total_return', 0):.2%}")
            print(f"   Sharpe ratio: {trading_results.get('sharpe_ratio', 0):.3f}")
            print(f"   Max drawdown: {trading_results.get('max_drawdown', 0):.2%}")
        
        # Check factor analysis
        factor_results = results.get('factor_analysis', {})
        if factor_results and 'pca_results' in factor_results:
            pca = factor_results['pca_results']
            if 'explained_variance' in pca:
                var = pca['explained_variance']
                print(f"\n🔍 Factor Analysis:")
                print(f"   First PC explains: {var.iloc[0]:.1%} of variance")
                print(f"   Cumulative variance: {var.cumsum().iloc[-1]:.1%}")
        
        # Check summary
        summary = filter_system.get_summary()
        perf = summary.get('performance_summary', {})
        print(f"\n📈 Performance Summary:")
        print(f"   Total Return: {perf.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
        print(f"   Total Trades: {perf.get('total_trades', 0)}")
        
        if perf.get('total_trades', 0) > 0:
            print("\n✅ SUCCESS: Signal generation is working!")
            print("   The system is now generating actual trades and meaningful metrics.")
        else:
            print("\n⚠️  WARNING: Still no trades generated.")
            print("   This may indicate further issues in the trading logic.")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    import pandas as pd
    success = test_signal_generation()
    sys.exit(0 if success else 1) 