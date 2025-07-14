#!/usr/bin/env python3
"""
Quick test of the enhanced Walleye case study system.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.equity_collector import EquityDataCollector
from src.data.news_collector import NewsDataCollector
from src.signals.multi_criteria_filter import MultiCriteriaFilter
from src.trading.contrarian_trader import ContrarianTrader

def quick_test():
    """Quick test of core components."""
    print("🚀 Quick Test of Enhanced Walleye System")
    print("=" * 50)
    
    try:
        # 1. Test equity data collection
        print("1. Testing equity data collection...")
        equity_collector = EquityDataCollector()
        equity_data = equity_collector.fetch_data(
            symbols=['XOP', 'SPY'],
            start_date='2023-01-01',
            end_date='2023-01-31'  # Just one month for speed
        )
        print(f"   ✅ Collected data for {len(equity_data)} symbols")
        
        # 2. Test news collection
        print("2. Testing news collection...")
        news_collector = NewsDataCollector()
        news_data = news_collector.collect_news(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            max_articles=50  # Reduced for speed
        )
        print(f"   ✅ Collected {len(news_data)} news articles")
        
        # 3. Test signal generation
        print("3. Testing signal generation...")
        signal_filter = MultiCriteriaFilter()
        config = {
            'sentiment_threshold': 0.2,
            'volatility_threshold': 0.03,
            'reliability_threshold': 0.4
        }
        signals = signal_filter.generate_signals(equity_data, news_data, config)
        print(f"   ✅ Generated {len(signals)} signals")
        
        # 4. Test trading strategy
        print("4. Testing trading strategy...")
        trader = ContrarianTrader()
        trading_results = trader.backtest_strategy(equity_data, signals, config)
        print(f"   ✅ Completed backtest")
        
        # 5. Show results
        print("\n📊 Quick Results Summary:")
        print(f"   - News Articles: {len(news_data)}")
        print(f"   - Trading Signals: {len(signals)}")
        print(f"   - Buy Signals: {len(signals[signals['signal_direction'] > 0]) if not signals.empty else 0}")
        print(f"   - Sell Signals: {len(signals[signals['signal_direction'] < 0]) if not signals.empty else 0}")
        
        if 'daily_returns' in trading_results:
            returns = trading_results['daily_returns']
            total_return = (1 + returns).prod() - 1
            print(f"   - Total Return: {total_return:.2%}")
        
        print("\n✅ Quick test completed successfully!")
        print("🎯 Core pipeline is working - Streamlit app should function properly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during quick test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1) 