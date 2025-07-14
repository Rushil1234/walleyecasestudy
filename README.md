# Smart Signal Filtering for Oil-Linked Equity (XOP)

## Overview

This system implements a contrarian trading strategy for oil-related stocks (XOP ETF) that filters out noise from geopolitical headlines and focuses on genuine market-moving events. The system uses advanced NLP, sentiment analysis, and AI agents to identify high-quality trading signals while avoiding false breakouts from panic-driven rallies.

[https://kakkadwalleyecasestudy.streamlit.app/
](url)

## Key Features

### Agentic AI System
- **Memory & Novelty Detection**: LLM-based agent that remembers past headlines, signals, and outcomes
- **Dynamic Knowledge Base**: Tracks what types of headlines moved XOP vs. which faded
- **Novelty Detection**: Flags truly unique geopolitical scenarios by comparing with stored patterns
- **Meta-Learning**: Adjusts filtering criteria based on historical performance

### Advanced Signal Filtering
- **Sentiment Volatility Tracking**: Monitors rate of change in aggregated sentiment to identify noise
- **Source Reliability Weighting**: Assigns credibility scores to news sources based on historical accuracy
- **Multi-Criteria Filtering**: Only trades when sentiment score, volatility, and source credibility meet thresholds
- **News Clustering**: Identifies market-moving events by clustering similar articles

### Explainability & Adaptive Learning
- **Chain-of-Thought Explanations**: LLM provides rationale for each classification decision
- **Audit Trail**: Complete logging of decision-making process for human review
- **Continual Learning**: Periodic re-calibration to stay current with market conditions
- **Bias Mitigation**: Multi-source validation and bias detection

```
Data Collection → NLP Classification →  Signal Generation → Trading Strategy → Risk Evaluation
```


## Installation

```bash
# Clone the repository
git clone <repository-url>
cd walleyecasestudy

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Project Structure

```
walleyecasestudy/
├── src/
│   ├── data/           # Data collection and processing
│   ├── models/         # ML models and sentiment analysis
│   ├── signals/        # Signal generation and filtering
│   ├── trading/        # Trading strategy implementation
│   ├── risk/           # Risk management and evaluation
│   └── agents/         # AI agent implementation
├── tests/              # Unit tests and validation
├── config/             # Configuration files
├── data/               # Data storage
└── docs/               # Documentation
```
## Configuration

The system is highly configurable through YAML files:

```yaml
# config/trading.yaml
trading:
  position_size: 0.02  # 2% of portfolio
  max_positions: 3
  stop_loss: 0.05      # 5% stop loss
  holding_period: 2    # days

# config/sentiment.yaml
sentiment:
  model: "llama2"
  threshold: 0.7
  volatility_window: 3
  min_sources: 3
```

## Risk Management

- **Lookahead Bias Prevention**: Strict timestamp validation
- **Overfitting Protection**: Walk-forward validation
- **Liquidity Constraints**: Position sizing based on ADV
- **Regime Testing**: Performance across different market conditions

## Performance Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Information Ratio**: Alpha relative to benchmark
- **Calmar Ratio**: Annual return / maximum drawdown

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for research and educational purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## References

- VanEck Vectors Oil Services ETF (XOP) research
- Academic literature on news sentiment and oil prices
- Contrarian trading strategies and mean reversion
- LLM-driven financial analysis and agentic systems 
