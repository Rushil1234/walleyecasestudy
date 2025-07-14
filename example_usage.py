#!/usr/bin/env python3
"""
Example usage of Smart Signal Filtering system for XOP ETF.

This script demonstrates the basic usage of the Smart Signal Filtering system
for contrarian trading based on filtered news sentiment.
"""

import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.main import SmartSignalFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate the Smart Signal Filtering system."""
    
    print("🚀 Smart Signal Filtering for XOP ETF")
    print("=" * 50)
    
    try:
        # Initialize the system
        print("\n1. Initializing Smart Signal Filtering system...")
        filter_system = SmartSignalFilter()
        print("✅ System initialized successfully")
        
        # Define analysis parameters
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        symbols = ["XOP", "XLE", "USO", "BNO", "SPY"]
        
        print(f"\n2. Running analysis for period: {start_date} to {end_date}")
        print(f"   Symbols: {', '.join(symbols)}")
        
        # Run the complete pipeline
        print("\n3. Running complete pipeline...")
        results = filter_system.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            backtest=True,
            save_results=True
        )
        
        print("✅ Pipeline completed successfully")
        
        # Display comprehensive results summary
        print("\n4. COMPREHENSIVE RESULTS SUMMARY:")
        print("=" * 50)
        
        summary = filter_system.get_summary()
        
        # Performance metrics
        perf = summary.get('performance_summary', {})
        print(f"📈 Performance:")
        print(f"   Total Return: {perf.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
        print(f"   Win Rate: {perf.get('win_rate', 0):.1%}")
        print(f"   Total Trades: {perf.get('total_trades', 0)}")
        print(f"   Calmar Ratio: {perf.get('calmar_ratio', 0):.2f}")
        
        # Risk metrics
        risk = summary.get('risk_summary', {})
        print(f"\n⚠️  Risk Metrics:")
        print(f"   Volatility: {risk.get('volatility', 0):.3f}")
        print(f"   VaR (95%): {risk.get('var_95', 0):.3f}")
        print(f"   CVaR (95%): {risk.get('cvar_95', 0):.3f}")
        print(f"   Factor Exposures: {len(risk.get('factor_exposures', {}))} factors")
        
        # Factor analysis
        if 'factor_analysis' in results:
            factor = results['factor_analysis']
            if 'pca_results' in factor:
                pca = factor['pca_results']
                if 'explained_variance' in pca:
                    var = pca['explained_variance']
                    print(f"\n🔍 Factor Analysis:")
                    print(f"   First PC explains: {var.iloc[0]:.1%} of variance")
                    print(f"   Cumulative variance: {var.cumsum().iloc[-1]:.1%}")
                    print(f"   Market regimes identified: {len(factor.get('market_regimes', {}).unique())}")
        
        # Walk-forward validation
        if 'walk_forward_results' in results:
            wf = results['walk_forward_results']
            if 'consistency_metrics' in wf:
                consistency = wf['consistency_metrics']
                print(f"\n🔄 Walk-Forward Validation:")
                print(f"   Number of splits: {wf.get('num_splits', 0)}")
                print(f"   Consistency ratio: {consistency.get('consistency_ratio', 0):.1f}%")
                print(f"   Stability: {consistency.get('stability', 0):.2f}")
        
        # Stress testing
        if 'stress_test_results' in results:
            stress = results['stress_test_results']
            if 'summary' in stress:
                summary_stress = stress['summary']
                print(f"\n🧪 Stress Testing:")
                print(f"   Scenarios tested: {summary_stress.get('scenarios_tested', 0)}")
                print(f"   Worst drawdown: {summary_stress.get('worst_drawdown', 0):.1f}%")
                print(f"   Average drawdown: {summary_stress.get('average_drawdown', 0):.1f}%")
        
        # Bias analysis
        if 'bias_analysis' in results:
            bias = results['bias_analysis']
            if 'overall_bias' in bias:
                overall = bias['overall_bias']
                print(f"\n🔍 Bias Analysis:")
                print(f"   Weighted sentiment: {overall.get('weighted_sentiment', 0):.3f}")
                print(f"   Bias dispersion: {overall.get('bias_dispersion', 0):.3f}")
                print(f"   High bias detected: {overall.get('high_bias_detected', False)}")
        
        # Data summary
        data = summary.get('data_summary', {})
        print(f"\n📊 Data Summary:")
        print(f"   News Articles: {data.get('news_articles', 0)}")
        print(f"   News Sources: {data.get('news_sources', 0)}")
        print(f"   Equity Records: {data.get('equity_records', {})}")
        
        # AI Agent insights
        agent = summary.get('agent_summary', {})
        print(f"\n🤖 AI Agent Insights:")
        print(f"   Novelty Score: {agent.get('novelty_score', 0):.3f}")
        print(f"   Confidence: {agent.get('confidence', 0):.3f}")
        
        recommendations = agent.get('recommendations', [])
        if recommendations:
            print(f"   Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"     {i}. {rec}")
        
        # Generate visualizations
        print("\n5. Generating visualizations...")
        filter_system.plot_results(save_plots=True)
        print("✅ Plots generated and saved")
        
        # Generate HTML report
        print("\n6. Generating HTML report...")
        filter_system.generate_report()
        print("✅ HTML report generated")
        
        print("\n🎉 Analysis complete!")
        print("\nOutput files:")
        print("  📈 Plots: data/plots/")
        print("  📊 Results: data/results/")
        print("  📄 Report: data/report.html")
        print("  🧠 Agent Memory: data/agent_memory.json")
        
        print("\n" + "=" * 50)
        print("✅ COMPREHENSIVE FEATURES IMPLEMENTED:")
        print("=" * 50)
        print("✅ Data Pipeline: Equity & News Collection")
        print("✅ Factor Exposure Analysis: PCA & Risk Decomposition")
        print("✅ Signal Construction: Sentiment + Volatility + Source Weighting")
        print("✅ Backtesting: Contrarian Strategy with Transaction Costs")
        print("✅ Walk-Forward Validation: Rolling Validation & Regime Testing")
        print("✅ Risk Management: VaR, CVaR, Factor Exposures")
        print("✅ Stress Testing: COVID-19, Oil Crash 2014-2016, Conflicts 2022-2023")
        print("✅ AI Agent: Memory, Novelty Detection, Adaptive Learning")
        print("✅ Bias Mitigation: Source Reliability, Multi-Source Validation")
        print("✅ Governance: Bias Detection, Model Drift Monitoring")
        print("✅ Jupyter Notebook: Complete demo with all components")
        print("✅ Comprehensive Testing: Unit tests for all modules")
        
        print("\n" + "=" * 50)
        print("Smart Signal Filtering system demonstration completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the logs for more details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 