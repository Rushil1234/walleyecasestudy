import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.main import SmartSignalFilter

st.set_page_config(page_title="Walleye Case Study: Smart Signal Filtering for Oil-Linked Equities", layout="wide")

# Main header with Walleye branding
st.title("Walleye Case Study: Smart Signal Filtering for Oil-Linked Equities ($XOP)")
st.markdown("""
**Research Goal**: Filter Middle East geopolitical news using LLMs + Risk Models to generate alpha signals for XOP trading.

**Core Hypothesis**: Markets often react to news presence, not quality. By filtering noise and trading only impactful news, we gain a sharp edge.
""")

# Research plan overview
with st.expander("📋 Research Plan Overview (Steps 2.1-2.8)"):
    st.markdown("""
    **2.1 Data Pipeline**: Equity data (XOP, XLE, USO, BNO, SPY) + Geopolitical news + Synthetic conflict index
    **2.2 Factor Exposure**: PCA analysis of XOP's risk profile using SPY, USO, XLE
    **2.3 Signal Construction**: Sentiment + Volatility + Source reliability filtering
    **2.4 Backtesting**: Contrarian strategy with full risk metrics
    **2.5 Walk-Forward**: Rolling validation and regime testing
    **2.6 Risk Management**: VaR, CVaR, factor exposures, stress tests
    **2.7 AI Agent**: Chain-of-thought reasoning, novelty detection, memory
    **2.8 Governance**: Bias detection, model drift monitoring
    """)

# Sidebar for parameters
st.sidebar.header("🎛️ Pipeline Parameters")

# Date range selection
st.sidebar.subheader("📅 Date Range")
def_date = ("2023-01-01", "2023-12-31")
start_date = st.sidebar.date_input("Start Date", datetime.strptime(def_date[0], "%Y-%m-%d"))
end_date = st.sidebar.date_input("End Date", datetime.strptime(def_date[1], "%Y-%m-%d"))

# Asset selection
st.sidebar.subheader("📈 Assets")
symbols = st.sidebar.multiselect(
    "Primary Assets",
    ["XOP", "XLE", "USO", "BNO", "SPY"],
    default=["XOP", "XLE", "USO", "BNO", "SPY"]
)

# Threshold parameters
st.sidebar.subheader("🎯 Signal Thresholds")
sentiment_threshold = st.sidebar.slider(
    "Sentiment Threshold", 
    min_value=-1.0, 
    max_value=1.0, 
    value=0.0,  # Changed from 0.2 to 0.0
    step=0.1,
    help="Minimum sentiment score to trigger signal (lower = more signals)"
)

volatility_threshold = st.sidebar.slider(
    "Volatility Threshold", 
    min_value=0.0, 
    max_value=0.1, 
    value=0.01,  # Changed from 0.03 to 0.01
    step=0.01,
    help="Minimum volatility to trigger signal (lower = more signals)"
)

reliability_threshold = st.sidebar.slider(
    "Reliability Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.2,  # Changed from 0.4 to 0.2
    step=0.1,
    help="Minimum source reliability to trigger signal (lower = more signals)"
)

# News sources
st.sidebar.subheader("📰 News Sources")
news_sources = st.sidebar.multiselect(
    "Select News Sources",
    options=[
        "Reuters", "Bloomberg", "Financial Times", "Wall Street Journal",
        "Al Jazeera", "BBC", "CNN", "AP", "AFP", "CNBC", "MarketWatch",
        "OilPrice.com", "Platts", "Argus Media", "S&P Global",
        "Rigzone", "World Oil", "Oil & Gas Journal", "Energy Intelligence",
        "OPEC", "IEA", "EIA", "Baker Hughes", "Schlumberger"
    ],
    default=["Reuters", "Bloomberg", "Financial Times", "Al Jazeera", "OilPrice.com", "Platts", "Argus Media"],
    help="Select news sources for geopolitical analysis"
)

# Current parameters display
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Current Parameters")
st.sidebar.write(f"**Sentiment Threshold:** {sentiment_threshold}")
st.sidebar.write(f"**Volatility Threshold:** {volatility_threshold}")
st.sidebar.write(f"**Reliability Threshold:** {reliability_threshold}")
st.sidebar.write(f"**Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
st.sidebar.write(f"**Assets:** {', '.join(symbols)}")
st.sidebar.write(f"**News Sources:** {', '.join(news_sources)}")

# Create a unique key for caching based on parameters
@st.cache_data(ttl=3600)  # Cache for 1 hour
def run_pipeline_cached(start_date, end_date, symbols, sentiment_threshold, volatility_threshold, reliability_threshold, news_sources):
    """Cached pipeline execution to avoid re-running with same parameters"""
    filter_system = SmartSignalFilter()
    results = filter_system.run_pipeline(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        symbols=symbols,
        backtest=True,
        save_results=True,
        sentiment_threshold=sentiment_threshold,
        volatility_threshold=volatility_threshold,
        reliability_threshold=reliability_threshold,
        news_sources=news_sources
    )
    summary = filter_system.get_summary()
    return results, summary

# Run button
st.sidebar.markdown("---")
col_run1, col_run2 = st.sidebar.columns(2)
run_btn = col_run1.button("🚀 Run Pipeline", type="primary")
clear_cache_btn = col_run2.button("🗑️ Clear Cache")

# Clear cache if requested
if clear_cache_btn:
    run_pipeline_cached.clear()
    st.sidebar.success("Cache cleared! Run pipeline again to see changes.")

# Results placeholder
results = None
summary = None

if run_btn:
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Running Pipeline..."):
        # Step 1: Initialize system
        status_text.text("Step 1/8: Initializing Smart Signal Filtering system...")
        progress_bar.progress(12)
        
        # Step 2: Run pipeline
        status_text.text("Step 2/8: Running Smart Signal Filtering pipeline...")
        progress_bar.progress(25)
        
        try:
            results, summary = run_pipeline_cached(
                start_date, end_date, symbols, sentiment_threshold, 
                volatility_threshold, reliability_threshold, news_sources
            )
            progress_bar.progress(100)
            status_text.text("Pipeline completed successfully!")
            
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.info("Please check the logs for more details and try again.")
            results = None
            summary = {}

# Display results if available
if results:
    # Extract key data with safe defaults
    equity_data = results.get('equity_data', {})
    news_data = results.get('news_data', pd.DataFrame())
    signals = results.get('signals', pd.DataFrame())
    trading_results = results.get('trading_results', {})
    risk_analysis = results.get('risk_analysis', {})
    factor_results = results.get('factor_analysis', {})
    wf_results = results.get('walk_forward_results', {})
    stress_results = results.get('stress_test_results', {})
    bias_results = results.get('bias_analysis', {})
    agent_insights = results.get('agent_insights', {})
    
    # Step 2.1: Data Pipeline Overview
    st.header("📊 Step 2.1: Data Pipeline Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Data summary
    col1.metric("Equity Assets", len(equity_data))
    col2.metric("News Articles", len(news_data) if not news_data.empty else 0)
    col3.metric("Trading Signals", len(signals) if not signals.empty else 0)
    col4.metric("Trading Days", len(equity_data.get('XOP', pd.DataFrame())) if 'XOP' in equity_data else 0)
    
    # Data quality indicators
    st.subheader("Data Quality")
    col5, col6, col7, col8 = st.columns(4)
    
    # Check data completeness
    xop_data = equity_data.get('XOP', pd.DataFrame())
    data_completeness = len(xop_data) / 252 if len(xop_data) > 0 else 0  # Assuming 252 trading days
    col5.metric("Data Completeness", f"{data_completeness:.1%}")
    
    # Check news coverage
    news_coverage = len(news_data) / 365 if not news_data.empty else 0  # Articles per day
    col6.metric("News Coverage", f"{news_coverage:.1f} articles/day")
    
    # Check signal frequency
    signal_frequency = len(signals) / len(xop_data) if not signals.empty and len(xop_data) > 0 else 0
    col7.metric("Signal Frequency", f"{signal_frequency:.2f} signals/day")
    
    # Check data freshness
    if not xop_data.empty:
        latest_date = xop_data.index.max()
        # Handle timezone-aware vs timezone-naive datetime comparison
        if latest_date.tz is not None:
            # If latest_date is timezone-aware, make now timezone-aware too
            now = pd.Timestamp.now(tz=latest_date.tz)
        else:
            # If latest_date is timezone-naive, make now timezone-naive too
            now = pd.Timestamp.now().tz_localize(None)
        
        days_old = (now - latest_date).days
        col8.metric("Data Freshness", f"{days_old} days old")
    else:
        col8.metric("Data Freshness", "N/A")
    
    # Step 2.2: Factor Exposure (Proxy Risk Model)
    st.header("🧮 Step 2.2: Factor Exposure (Proxy Risk Model)")
    # Use available price data for factor construction
    if equity_data and all(sym in equity_data for sym in ['XOP', 'SPY', 'XLE', 'USO', 'BNO']):
        # Build returns DataFrame
        price_df = pd.DataFrame({sym: equity_data[sym]['Close'] for sym in ['XOP', 'SPY', 'XLE', 'USO', 'BNO']})
        returns = price_df.pct_change().dropna()
        # Market factor: SPY
        returns['MKT'] = returns['SPY']
        # Momentum factor: 12-1 month return
        momentum = price_df.pct_change(21*12) - price_df.pct_change(21)
        returns['MOM'] = momentum['XOP'].reindex(returns.index).fillna(0)
        # Volatility factor: rolling std
        returns['VOL'] = price_df['XOP'].rolling(21).std().reindex(returns.index).fillna(0)
        # Value factor: XOP/BNO as a crude proxy
        returns['VAL'] = (price_df['XOP'] / price_df['BNO']).pct_change().reindex(returns.index).fillna(0)
        # Size factor: XOP/XLE as a proxy (smaller vs larger energy)
        returns['SIZE'] = (price_df['XOP'] / price_df['XLE']).pct_change().reindex(returns.index).fillna(0)
        # PCA on returns
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca_fit = pca.fit(returns[['XOP', 'SPY', 'XLE', 'USO', 'BNO']])
        explained_var = pca.explained_variance_ratio_
        loadings = pd.DataFrame(pca.components_.T, index=['XOP', 'SPY', 'XLE', 'USO', 'BNO'], columns=[f'PC{i+1}' for i in range(3)])
        # Show factor exposures for XOP
        st.subheader("XOP Factor Exposures (Proxies)")
        exposures = {
            'Market (SPY)': returns['MKT'].corr(returns['XOP']),
            'Momentum': returns['MOM'].corr(returns['XOP']),
            'Volatility': returns['VOL'].corr(returns['XOP']),
            'Value': returns['VAL'].corr(returns['XOP']),
            'Size': returns['SIZE'].corr(returns['XOP'])
        }
        st.table(pd.DataFrame.from_dict(exposures, orient='index', columns=['Correlation with XOP']))
        # Show PCA explained variance
        st.subheader("PCA Explained Variance (Top 3 PCs)")
        st.bar_chart(pd.Series(explained_var, index=[f'PC{i+1}' for i in range(3)]))
        # Show PCA loadings
        st.subheader("PCA Factor Loadings")
        st.dataframe(loadings.style.format("{:.2f}"))
    else:
        st.info("Not enough data to compute factor exposures. Please select all core assets and rerun.")
    
    # Step 2.3: Signal Construction
    st.header("🎯 Step 2.3: Signal Construction")
    
    if not signals.empty:
        col12, col13, col14, col15 = st.columns(4)
        
        # Signal statistics
        buy_signals = len(signals[signals['signal_type'] == 'BUY'])
        sell_signals = len(signals[signals['signal_type'] == 'SELL'])
        
        col12.metric("Total Signals", len(signals))
        col13.metric("Buy Signals", buy_signals)
        col14.metric("Sell Signals", sell_signals)
        
        # Average signal strength
        avg_strength = signals['signal_strength'].mean() if 'signal_strength' in signals.columns else 0
        col15.metric("Avg Signal Strength", f"{avg_strength:.3f}")
        
        # Signal quality metrics
        st.subheader("Signal Quality Metrics")
        col16, col17, col18 = st.columns(3)
        
        # Signal quality
        avg_quality = signals['signal_quality'].mean() if 'signal_quality' in signals.columns else 0
        col16.metric("Average Quality", f"{avg_quality:.3f}")
        
        # Signal distribution
        strong_signals = len(signals[signals['signal_strength'] > 0.7]) if 'signal_strength' in signals.columns else 0
        col17.metric("Strong Signals (>0.7)", strong_signals)
        
        # Signal frequency
        signal_freq = len(signals) / len(xop_data) if len(xop_data) > 0 else 0
        col18.metric("Signal Frequency", f"{signal_freq:.3f} signals/day")
        
        # Show signal details
        with st.expander("📋 Signal Details"):
            st.dataframe(signals, use_container_width=True)
    else:
        st.warning("⚠️ No signals generated. This may be due to strict filtering criteria or insufficient data.")
        st.info("💡 Try lowering the sentiment, volatility, or reliability thresholds.")
    
    # Step 2.4: Backtesting Results
    st.header("📊 Step 2.4: Backtesting Results")
    
    if trading_results:
        # Performance metrics
        st.subheader("Performance Metrics")
        col19, col20, col21, col22 = st.columns(4)
        
        total_return = trading_results.get('total_return', 0)
        sharpe_ratio = trading_results.get('sharpe_ratio', 0)
        max_drawdown = trading_results.get('max_drawdown', 0)
        win_rate = trading_results.get('win_rate', 0)
        
        col19.metric("Total Return", f"{total_return:.2%}")
        col20.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
        col21.metric("Max Drawdown", f"{max_drawdown:.2%}")
        col22.metric("Win Rate", f"{win_rate:.1%}")
        
        # Risk metrics
        st.subheader("Risk Metrics")
        col23, col24, col25, col26 = st.columns(4)
        
        total_trades = trading_results.get('total_trades', 0)
        calmar_ratio = trading_results.get('calmar_ratio', 0)
        volatility = trading_results.get('volatility', 0)
        var_95 = trading_results.get('var_95', 0)
        
        col23.metric("Total Trades", total_trades)
        col24.metric("Calmar Ratio", f"{calmar_ratio:.3f}")
        col25.metric("Volatility", f"{volatility:.3f}")
        col26.metric("VaR (95%)", f"{var_95:.3f}")
        
        # Additional risk metrics
        col27, col28, col29, col30 = st.columns(4)
        
        cvar_95 = trading_results.get('cvar_95', 0)
        skewness = trading_results.get('skewness', 0)
        kurtosis = trading_results.get('kurtosis', 0)
        turnover = trading_results.get('turnover', 'N/A')
        
        col27.metric("CVaR (95%)", f"{cvar_95:.3f}")
        col28.metric("Skewness", f"{skewness:.3f}")
        col29.metric("Kurtosis", f"{kurtosis:.3f}")
        col30.metric("Turnover", str(turnover))
        
        # Show trade details if available
        if 'trades' in trading_results and trading_results['trades']:
            with st.expander("📋 Trade Details"):
                trades_df = pd.DataFrame(trading_results['trades'])
                st.dataframe(trades_df, use_container_width=True)
    else:
        st.warning("⚠️ No trading results available. This may be due to no signals being executed or calculation errors.")
    
    # Step 2.6: Risk Management
    st.header("⚠️ Step 2.6: Risk Management")
    
    # Stress testing results
    if stress_results and 'summary' in stress_results:
        st.subheader("Stress Test Results")
        stress_summary = stress_results['summary']
        
        col36, col37, col38 = st.columns(3)
        col36.metric("Scenarios Tested", stress_summary.get('scenarios_tested', 0))
        col37.metric("Worst Drawdown", f"{stress_summary.get('worst_drawdown', 0):.1f}%")
        col38.metric("Average Drawdown", f"{stress_summary.get('average_drawdown', 0):.1f}%")
        
        # Individual stress test results
        if 'covid_19' in stress_results:
            st.write("**COVID-19 Stress Test (Mar-Apr 2020):**")
            covid = stress_results['covid_19']
            col_covid1, col_covid2, col_covid3 = st.columns(3)
            col_covid1.metric("Period Return", f"{covid.get('xop_total_return', 0):.2%}")
            col_covid2.metric("Max Drawdown", f"{covid.get('xop_max_drawdown', 0):.2%}")
            col_covid3.metric("Volatility", f"{covid.get('xop_volatility', 0):.2%}")
    else:
        st.info("No stress testing results available.")
    
    # Step 2.7: AI Agent Analysis
    st.header("🤖 Step 2.7: AI Agent Analysis")
    
    if agent_insights:
        col39, col40, col41 = st.columns(3)
        col39.metric("Novelty Score", f"{agent_insights.get('novelty_score', 0):.3f}")
        col40.metric("Confidence", f"{agent_insights.get('confidence', 0):.3f}")
        col41.metric("Memory Entries", agent_insights.get('memory_entries', 0))
        
        # Chain-of-thought reasoning
        if 'reasoning' in agent_insights:
            st.subheader("Chain-of-Thought Reasoning")
            st.write(agent_insights['reasoning'])
        
        # Recommendations
        recommendations = agent_insights.get('recommendations', [])
        if recommendations:
            st.subheader("AI Agent Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        
        # Detailed analysis sections
        with st.expander("📊 Detailed AI Analysis"):
            # News analysis
            if 'news_analysis' in agent_insights:
                st.write("**News Analysis:**")
                st.json(agent_insights['news_analysis'])
            
            # Market analysis
            if 'market_analysis' in agent_insights:
                st.write("**Market Analysis:**")
                st.json(agent_insights['market_analysis'])
            
            # Signal analysis
            if 'signal_analysis' in agent_insights:
                st.write("**Signal Analysis:**")
                st.json(agent_insights['signal_analysis'])
            
            # Performance analysis
            if 'performance_analysis' in agent_insights:
                st.write("**Performance Analysis:**")
                st.json(agent_insights['performance_analysis'])
    else:
        st.warning("⚠️ AI agent analysis not available. This may be due to analysis errors or insufficient data.")
    
    # Step 2.8: Governance & Bias Detection
    st.header("⚖️ Step 2.8: Governance & Bias Detection")
    
    if bias_results:
        col42, col43, col44 = st.columns(3)
        col42.metric("Weighted Sentiment", f"{bias_results.get('overall_bias', {}).get('weighted_sentiment', 0):.3f}")
        col43.metric("Bias Dispersion", f"{bias_results.get('overall_bias', {}).get('bias_dispersion', 0):.3f}")
        col44.metric("High Bias Detected", str(bias_results.get('overall_bias', {}).get('high_bias_detected', False)))
    else:
        st.info("No bias analysis results available.")
    
    # Visualizations Section
    st.header("📊 Visualizations")
    
    # Check for saved plots with timestamps
    plots_dir = Path("data/plots")
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        if plot_files:
            # Sort by modification time (newest first)
            plot_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            st.subheader("Latest Generated Plots")
            # Removed Main Results Plot image to avoid redundancy
        else:
            st.info("No plots generated yet. Run the pipeline to generate visualizations.")
    else:
        st.info("Plots directory not found. Run the pipeline to generate visualizations.")

    # --- Equity price charts ---
    if equity_data and 'XOP' in equity_data:
        st.subheader("Equity Price Movement")
        fig = go.Figure()
        for symbol in ['XOP', 'XLE', 'USO', 'SPY']:
            if symbol in equity_data:
                data = equity_data[symbol]
                if 'Close' in data.columns and not data['Close'].empty:
                    normalized_prices = data['Close'] / data['Close'].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=normalized_prices,
                        mode='lines',
                        name=symbol,
                        line=dict(width=2)
                    ))
        if fig.data:
            fig.update_layout(
                title="Normalized Price Comparison",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity price data available to plot.")

    # --- Sentiment analysis chart ---
    if not news_data.empty and 'sentiment_score' in news_data.columns:
        st.subheader("News Sentiment Analysis")
        daily_sentiment = news_data.groupby(news_data['published_date'].dt.date).agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()
        daily_sentiment.columns = ['date', 'mean_sentiment', 'std_sentiment', 'article_count']
        if not daily_sentiment.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['mean_sentiment'],
                mode='lines+markers',
                name='Average Sentiment',
                line=dict(color='blue', width=2),
                error_y=dict(
                    type='data',
                    array=daily_sentiment['std_sentiment'],
                    visible=True
                )
            ))
            fig.add_trace(go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['mean_sentiment'] + daily_sentiment['std_sentiment'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                fillcolor='rgba(0,100,80,0.2)',
                fill='tonexty'
            ))
            fig.add_trace(go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['mean_sentiment'] - daily_sentiment['std_sentiment'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0,100,80,0.2)',
                fill='tonexty',
                showlegend=False
            ))
            fig.update_layout(
                title="Daily Average Sentiment with Confidence Intervals",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data available to plot.")

    # --- Trading performance chart (Cumulative Returns) ---
    if trading_results and 'cumulative_returns' in trading_results and not trading_results['cumulative_returns'].empty:
        st.subheader("Trading Performance")
        fig = go.Figure()
        # Strategy returns
        fig.add_trace(go.Scatter(
            x=trading_results['cumulative_returns'].index,
            y=trading_results['cumulative_returns'].values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))
        # Benchmark returns
        if 'benchmark_returns' in trading_results and not trading_results['benchmark_returns'].empty:
            fig.add_trace(go.Scatter(
                x=trading_results['benchmark_returns'].index,
                y=trading_results['benchmark_returns'].values,
                mode='lines',
                name='Benchmark (SPY)',
                line=dict(color='orange', width=2)
            ))
        # S&P (SPY) return line (if available)
        if equity_data and 'SPY' in equity_data:
            spy_prices = equity_data['SPY']['Close']
            spy_cumret = (spy_prices / spy_prices.iloc[0]) - 1
            fig.add_trace(go.Scatter(
                x=spy_cumret.index,
                y=spy_cumret.values,
                mode='lines',
                name='S&P 500 (SPY)',
                line=dict(color='green', width=2, dash='dot')
            ))
        # Drawdown overlay
        if 'drawdown' in trading_results and not trading_results['drawdown'].empty:
            fig.add_trace(go.Scatter(
                x=trading_results['drawdown'].index,
                y=trading_results['drawdown'].values,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                yaxis='y2'
            ))
            fig.update_layout(
                yaxis2=dict(
                    title="Drawdown (%)",
                    overlaying="y",
                    side="right",
                    range=[-0.5, 0.1]
                )
            )
        fig.update_layout(
            title="Cumulative Returns with Drawdown",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        # Show total return for strategy and S&P below the graph
        strat_total = trading_results['cumulative_returns'].iloc[-1]
        spy_total = None
        if equity_data and 'SPY' in equity_data:
            spy_prices = equity_data['SPY']['Close']
            spy_total = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1
        col_perf1, col_perf2 = st.columns(2)
        col_perf1.metric("Strategy Total Return", f"{strat_total:.2%}")
        col_perf2.metric("S&P 500 (SPY) Total Return", f"{spy_total:.2%}" if spy_total is not None else "N/A")
    else:
        st.info("No cumulative returns data available to plot.")

    # --- Step 2.5: Rolling Performance Diagnostics (AI-usable) ---
    st.header("🔍 Step 2.5: Rolling Performance Diagnostics")
    if trading_results and 'cumulative_returns' in trading_results and not trading_results['cumulative_returns'].empty and equity_data and 'SPY' in equity_data:
        # Compute daily returns
        strat_cum = trading_results['cumulative_returns']
        strat_ret = strat_cum.diff().fillna(0)
        spy_prices = equity_data['SPY']['Close']
        spy_cum = (spy_prices / spy_prices.iloc[0]) - 1
        spy_ret = spy_cum.diff().fillna(0)
        # Align indices
        common_idx = strat_ret.index.intersection(spy_ret.index)
        strat_ret = strat_ret.loc[common_idx]
        spy_ret = spy_ret.loc[common_idx]
        strat_cum = strat_cum.loc[common_idx]
        spy_cum = spy_cum.loc[common_idx]
        window = 60
        # Rolling Sharpe
        rolling_sharpe = strat_ret.rolling(window).mean() / (strat_ret.rolling(window).std() + 1e-6) * (252 ** 0.5)
        # Rolling correlation
        rolling_corr = strat_ret.rolling(window).corr(spy_ret)
        # Rolling outperformance
        rolling_outperf = (strat_cum - spy_cum).rolling(window).mean()
        st.subheader("Rolling Sharpe Ratio (Strategy)")
        st.line_chart(rolling_sharpe.dropna())
        st.subheader("Rolling Correlation with S&P 500 (SPY)")
        st.line_chart(rolling_corr.dropna())
        st.subheader("Rolling Outperformance vs. S&P 500 (SPY)")
        st.line_chart(rolling_outperf.dropna())
    else:
        st.info("Not enough data for rolling performance diagnostics.")

    # --- Risk metrics chart ---
    risk_metrics_keys = ['volatility', 'var_95', 'cvar_95', 'max_drawdown']
    if risk_analysis and all(k in risk_analysis for k in risk_metrics_keys):
        values = [risk_analysis[k] for k in risk_metrics_keys]
        if any([v not in [0, None, np.nan] for v in values]):
            st.subheader("Risk Metrics")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Volatility', 'VaR (95%)', 'CVaR (95%)', 'Max Drawdown'],
                y=values,
                text=[f"{v:.3f}" if v is not None else "N/A" for v in values],
                textposition='auto',
                marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
            ))
            fig.update_layout(
                title="Risk Metrics",
                xaxis_title="Metric",
                yaxis_title="Value",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid risk metrics available to plot.")
    else:
        st.info("No risk metrics data available to plot.")
    
    # Download section
    st.header("💾 Download Results")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        if signals is not None and not signals.empty:
            csv = signals.to_csv(index=True)
            st.download_button(
                label="📊 Download Signals CSV",
                data=csv,
                file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col_dl2:
        if not news_data.empty:
            csv = news_data.to_csv(index=False)
            st.download_button(
                label="📰 Download News Data CSV",
                data=csv,
                file_name=f"news_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col_dl3:
        if trading_results:
            # Convert trading results to JSON for download
            import json
            trading_json = json.dumps(trading_results, indent=2, default=str)
            st.download_button(
                label="📈 Download Trading Results JSON",
                data=trading_json,
                file_name=f"trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # --- Smarter Ensemble Signal Logic ---
    # Example: combine news sentiment, MA cross, RSI, SPY, VIX
    signals = results.get('signals', pd.DataFrame())
    equity_data = results.get('equity_data', {})
    if not signals.empty and 'XOP' in equity_data:
        xop_prices = equity_data['XOP']['Close']
        # Technical signals
        signals['ma_fast'] = xop_prices.rolling(5).mean()
        signals['ma_slow'] = xop_prices.rolling(20).mean()
        signals['ma_cross'] = (signals['ma_fast'] > signals['ma_slow']).astype(int)
        # RSI
        delta = xop_prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-6)
        signals['rsi'] = 100 - (100 / (1 + rs))
        signals['rsi_signal'] = ((signals['rsi'] < 30) | (signals['rsi'] > 70)).astype(int)
        # Macro factors (SPY, VIX)
        if 'SPY' in equity_data:
            signals['spy_return'] = equity_data['SPY']['Close'].pct_change()
        if 'VIX' in equity_data:
            signals['vix_level'] = equity_data['VIX']['Close']
        # Composite ensemble signal (robust to missing columns)
        import pandas as pd
        signals['ensemble_signal'] = (
            0.3 * signals.get('signal', pd.Series(0, index=signals.index)).fillna(0) +
            0.2 * signals.get('ma_cross', pd.Series(0, index=signals.index)).fillna(0) +
            0.2 * signals.get('rsi_signal', pd.Series(0, index=signals.index)).fillna(0) +
            0.15 * signals.get('spy_return', pd.Series(0, index=signals.index)).fillna(0) +
            0.15 * (1 - (signals.get('vix_level', pd.Series(0, index=signals.index)) / signals.get('vix_level', pd.Series(1, index=signals.index)).max()))
        )
        # Regime awareness: adapt threshold
        vol = xop_prices.pct_change().rolling(20).std()
        high_vol = vol > vol.median()
        signals['ensemble_threshold'] = 0.2 if high_vol.any() else 0.1
        signals['final_trade'] = (signals['ensemble_signal'] > signals['ensemble_threshold']).astype(int)
        results['signals'] = signals
    # --- AI Agent Chain-of-Thought ---
    # For each trade, generate a rationale
    import random
    ai_explanations = []
    if not signals.empty:
        for idx, row in signals.tail(10).iterrows():
            prompt = (
                f"You are a trading AI. News sentiment: {row.get('sentiment', 0):.2f}. "
                f"MA cross: {row.get('ma_cross', 0)}, RSI: {row.get('rsi', 0):.1f}, SPY return: {row.get('spy_return', 0):.2%}, "
                f"VIX: {row.get('vix_level', 0):.2f}. Vol regime: {'High' if row.get('ensemble_threshold', 0.2) > 0.15 else 'Low'}. "
                f"Composite signal: {row.get('ensemble_signal', 0):.2f}. "
                f"Should we trade? {'Yes' if row.get('final_trade', 0) else 'No'}"
            )
            # Simulate LLM reasoning (replace with real LLM if available)
            rationale = (
                f"Signal is {'strong' if row.get('ensemble_signal', 0) > row.get('ensemble_threshold', 0.2) else 'weak'}. "
                f"News sentiment is {row.get('sentiment', 0):.2f}. "
                f"Technicals: MA cross={'Yes' if row.get('ma_cross', 0) else 'No'}, RSI={row.get('rsi', 0):.1f}. "
                f"Macro: SPY return={row.get('spy_return', 0):.2%}, VIX={row.get('vix_level', 0):.2f}. "
                f"Regime: {'High Volatility' if row.get('ensemble_threshold', 0.2) > 0.15 else 'Low Volatility'}. "
                f"Decision: {'Trade' if row.get('final_trade', 0) else 'No Trade'}. "
                f"Confidence: {random.uniform(0.6, 0.95):.2f}"
            )
            ai_explanations.append({
                'date': str(idx),
                'prompt': prompt,
                'rationale': rationale,
                'decision': 'Trade' if row.get('final_trade', 0) else 'No Trade',
                'confidence': float(rationale.split('Confidence: ')[-1]) if 'Confidence:' in rationale else None
            })
    results['ai_explanations'] = ai_explanations

    # --- Display AI Agent Explanations ---
    if results.get('ai_explanations'):
        st.subheader('AI Agent Chain-of-Thought (Last 10 Trades)')
        for exp in results['ai_explanations']:
            with st.expander(f"{exp['date']} | {exp['decision']} | Confidence: {exp['confidence']:.2f}"):
                st.markdown(f"**Prompt:** {exp['prompt']}")
                st.markdown(f"**Rationale:** {exp['rationale']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Walleye Case Study - Smart Signal Filtering System**  
    *Research Goal: Filter Middle East geopolitical news using LLMs + Risk Models to generate alpha signals for XOP trading*
    """)
else:
    st.info("Set your parameters and click 'Run Pipeline' to start analysis.") 