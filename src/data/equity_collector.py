"""
Equity data collection module for fetching ETF price data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import pickle
import gzip

logger = logging.getLogger(__name__)


class EquityDataCollector:
    """
    Collects equity data for XOP and related ETFs from various sources.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the equity data collector.
        
        Args:
            cache_dir: Directory to cache data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Default symbols to track
        self.symbols = {
            "XOP": "VanEck Vectors Oil Services ETF",
            "XLE": "Energy Select Sector SPDR Fund", 
            "USO": "United States Oil Fund",
            "BNO": "United States Brent Oil Fund",
            "SPY": "SPDR S&P 500 ETF",
            "OIH": "VanEck Vectors Oil Services ETF",
            "XES": "SPDR S&P Oil & Gas Equipment & Services ETF",
            "IEZ": "iShares U.S. Oil Equipment & Services ETF"
        }
        
    def fetch_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch equity data for given symbols and date range.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cached_data = self._load_from_cache(symbol, start_date, end_date, interval)
                if cached_data is not None:
                    data[symbol] = cached_data
                    logger.info(f"Loaded {symbol} data from cache")
                    continue
                
                # Fetch from Yahoo Finance
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Clean and process data
                df = self._clean_data(df)
                
                # Cache the data
                self._save_to_cache(df, symbol, start_date, end_date, interval)
                
                data[symbol] = df
                logger.info(f"Fetched {len(df)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
                
        return data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process raw equity data.
        
        Args:
            df: Raw DataFrame from yfinance
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with missing data
        df = df.dropna()
        
        # Calculate additional metrics
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Calculate technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            window: RSI window
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_cache_path(self, symbol: str, start_date: str, end_date: str, interval: str) -> Path:
        """
        Get cache file path for given parameters.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Cache file path
        """
        filename = f"{symbol}_{start_date}_{end_date}_{interval}.pkl.gz"
        return self.cache_dir / filename
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str, interval: str):
        """
        Save data to cache.
        
        Args:
            df: DataFrame to cache
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date, interval)
        
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.debug(f"Cached data for {symbol}")
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {e}")
    
    def _load_from_cache(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and fresh.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Cached DataFrame or None
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date, interval)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is fresh (less than 1 day old for daily data)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(days=1):
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                df = pickle.load(f)
            return df
        except Exception as e:
            logger.error(f"Error loading cache for {symbol}: {e}")
            return None
    
    def get_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate correlation matrix for returns.
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Correlation matrix
        """
        returns_data = {}
        
        for symbol, df in data.items():
            if 'Returns' in df.columns:
                returns_data[symbol] = df['Returns']
        
        if not returns_data:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()
    
    def calculate_factor_exposures(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculate factor exposures for each asset.
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Dictionary of factor exposures
        """
        exposures = {}
        
        # Define factors
        factors = {
            'Market': 'SPY',
            'Energy': 'XLE', 
            'Oil': 'USO'
        }
        
        for symbol, df in data.items():
            if symbol in factors.values():
                continue
                
            exposures[symbol] = {}
            
            for factor_name, factor_symbol in factors.items():
                if factor_symbol in data:
                    # Calculate beta to factor
                    asset_returns = df['Returns'].dropna()
                    factor_returns = data[factor_symbol]['Returns'].dropna()
                    
                    # Align dates
                    common_dates = asset_returns.index.intersection(factor_returns.index)
                    if len(common_dates) > 30:  # Need sufficient data
                        asset_returns = asset_returns.loc[common_dates]
                        factor_returns = factor_returns.loc[common_dates]
                        
                        # Calculate beta
                        covariance = np.cov(asset_returns, factor_returns)[0, 1]
                        factor_variance = np.var(factor_returns)
                        
                        if factor_variance > 0:
                            beta = covariance / factor_variance
                            exposures[symbol][factor_name] = beta
                        else:
                            exposures[symbol][factor_name] = 0.0
                    else:
                        exposures[symbol][factor_name] = 0.0
                else:
                    exposures[symbol][factor_name] = 0.0
        
        return exposures
    
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest closing prices for symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary of latest prices
        """
        prices = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if 'regularMarketPrice' in info and info['regularMarketPrice']:
                    prices[symbol] = info['regularMarketPrice']
                else:
                    # Fallback to historical data
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        prices[symbol] = hist['Close'].iloc[-1]
            except Exception as e:
                logger.error(f"Error getting latest price for {symbol}: {e}")
                continue
        
        return prices 