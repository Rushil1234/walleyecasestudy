"""
Basic functionality tests for Smart Signal Filtering system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.equity_collector import EquityDataCollector
from data.news_collector import NewsDataCollector
from models.sentiment_analyzer import LLMSentimentAnalyzer
from signals.multi_criteria_filter import MultiCriteriaFilter
from trading.contrarian_trader import ContrarianTrader
from risk.risk_manager import RiskManager
from agents.ai_agent import AIAgent


class TestSmartSignalFiltering(unittest.TestCase):
    """Test cases for Smart Signal Filtering system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_date = "2023-01-01"
        self.end_date = "2023-12-31"
        self.symbols = ["XOP", "SPY"]
        
    def test_equity_data_collector(self):
        """Test equity data collection."""
        collector = EquityDataCollector()
        data = collector.fetch_data(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 0)
        
        for symbol, df in data.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            self.assertIn('Close', df.columns)
            self.assertIn('Returns', df.columns)
    
    def test_news_data_collector(self):
        """Test news data collection."""
        collector = NewsDataCollector()
        data = collector.collect_news(
            start_date=datetime.strptime(self.start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(self.end_date, "%Y-%m-%d"),
            max_articles=10
        )
        
        self.assertIsInstance(data, pd.DataFrame)
        # Note: May be empty if no news sources are available
    
    def test_sentiment_analyzer(self):
        """Test sentiment analysis."""
        analyzer = LLMSentimentAnalyzer()
        
        # Create test data
        test_news = pd.DataFrame({
            'title': ['Test headline 1', 'Test headline 2'],
            'source': ['test_source'] * 2,
            'published_date': [datetime.now()] * 2,
            'summary': ['Test summary 1', 'Test summary 2']
        })
        
        result = analyzer.analyze_batch(test_news)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(test_news))
    
    def test_signal_filter(self):
        """Test signal generation."""
        filter_obj = MultiCriteriaFilter()
        
        # Create test equity data
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        test_equity_data = {
            'XOP': pd.DataFrame({
                'Close': np.random.randn(len(dates)).cumsum() + 100,
                'Returns': np.random.randn(len(dates)) * 0.02,
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
        }
        
        # Create test news data
        test_news_data = pd.DataFrame({
            'title': ['Test news'] * 10,
            'source': ['test'] * 10,
            'published_date': dates[:10],
            'sentiment_score': np.random.randn(10) * 0.5,
            'reliability': [0.8] * 10
        })
        
        # Test config
        config = {
            'assets': {'primary': 'XOP'},
            'trading': {
                'contrarian_threshold': 0.5,
                'volatility_threshold': 0.05
            }
        }
        
        signals = filter_obj.generate_signals(test_equity_data, test_news_data, config)
        self.assertIsInstance(signals, pd.DataFrame)
    
    def test_contrarian_trader(self):
        """Test contrarian trading strategy."""
        trader = ContrarianTrader()
        
        # Create test data
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        test_equity_data = {
            'XOP': pd.DataFrame({
                'Close': np.random.randn(len(dates)).cumsum() + 100,
                'Returns': np.random.randn(len(dates)) * 0.02
            }, index=dates),
            'SPY': pd.DataFrame({
                'Close': np.random.randn(len(dates)).cumsum() + 400,
                'Returns': np.random.randn(len(dates)) * 0.015
            }, index=dates)
        }
        
        # Create test signals
        test_signals = pd.DataFrame({
            'signal_direction': [1, -1, 1],
            'signal_strength': [0.8, 0.7, 0.9]
        }, index=dates[:3])
        
        config = {
            'assets': {'primary': 'XOP'},
            'trading': {
                'position_size': 0.02,
                'max_positions': 3
            }
        }
        
        results = trader.backtest_strategy(test_equity_data, test_signals, config)
        self.assertIsInstance(results, dict)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
    
    def test_risk_manager(self):
        """Test risk analysis."""
        risk_manager = RiskManager()
        
        # Create test trading results
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        test_trading_results = {
            'daily_returns': pd.Series(np.random.randn(len(dates)) * 0.02, index=dates)
        }
        
        # Create test equity data
        test_equity_data = {
            'SPY': pd.DataFrame({
                'Returns': np.random.randn(len(dates)) * 0.015
            }, index=dates)
        }
        
        config = {'trading': {}}
        
        risk_analysis = risk_manager.analyze_risk(test_trading_results, test_equity_data, config)
        self.assertIsInstance(risk_analysis, dict)
        self.assertIn('volatility', risk_analysis)
        self.assertIn('var_95', risk_analysis)
    
    def test_ai_agent(self):
        """Test AI agent."""
        agent = AIAgent()
        
        # Create test data
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        test_equity_data = {
            'XOP': pd.DataFrame({
                'Returns': np.random.randn(len(dates)) * 0.02
            }, index=dates)
        }
        
        test_news_data = pd.DataFrame({
            'sentiment_score': np.random.randn(10) * 0.5
        })
        
        test_signals = pd.DataFrame({
            'signal_quality': [0.8, 0.7, 0.9]
        }, index=dates[:3])
        
        test_trading_results = {'total_return': 0.05}
        
        insights = agent.analyze_market(
            test_equity_data, test_news_data, test_signals, test_trading_results
        )
        self.assertIsInstance(insights, dict)
        self.assertIn('novelty_score', insights)
        self.assertIn('confidence', insights)
        self.assertIn('recommendations', insights)


if __name__ == '__main__':
    unittest.main() 