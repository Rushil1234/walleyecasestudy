"""
Factor Exposure Analysis Module

Implements PCA, risk decomposition, and factor modeling for XOP and related ETFs.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import logging
import datetime
try:
    import pandas_datareader.data as web
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

logger = logging.getLogger(__name__)


class FactorExposureAnalyzer:
    """
    Analyzes factor exposures for XOP and related ETFs using PCA and risk decomposition.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize factor analyzer.
        
        Args:
            symbols: List of ETF symbols to analyze
        """
        self.symbols = symbols or ['XOP', 'XLE', 'USO', 'BNO', 'SPY']
        self.pca = None
        self.scaler = StandardScaler()
        self.factor_loadings = None
        self.explained_variance = None
    # 1️⃣  fetch_factor_data  – one combined download + softer NA rule
    def fetch_factor_data(self, start_date: str, end_date: str):
        """
        Fetch daily returns and volume for factor analysis.
        Returns:
            returns_df: DataFrame with daily returns
            volume_df: DataFrame with daily volume
        """
        logger.info(f"Fetching factor data for {self.symbols}")
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        today = datetime.date.today().strftime('%Y-%m-%d')
        if end_date > today:
            logger.warning(f"End date {end_date} is in the future. Clipping to today: {today}")
            end_date = today
        returns_data = {}
        volume_data = {}
        missing_symbols = []
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                data.index = pd.to_datetime(data.index)
                if data.empty:
                    logger.warning(f"No data for symbol {symbol} in range {start_date} to {end_date}")
                    missing_symbols.append(symbol)
                    continue
                returns_data[symbol] = data['Close'].pct_change()
                volume_data[symbol] = data['Volume']
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                missing_symbols.append(symbol)
                continue
        if missing_symbols:
            logger.warning(f"Missing data for symbols: {missing_symbols}")
        if not returns_data:
            logger.error("No data could be fetched for any symbols. Aborting analysis.")
            return pd.DataFrame(), pd.DataFrame()
        returns_df = pd.DataFrame(returns_data).dropna(axis=1, how='all').dropna(axis=0, how='all')
        volume_df = pd.DataFrame(volume_data).dropna(axis=1, how='all').dropna(axis=0, how='all')
        if len(returns_df) < 30:
            logger.error(f"Insufficient data after cleaning: only {len(returns_df)} rows. Need at least 30 for rolling calculations.")
            return pd.DataFrame(), pd.DataFrame()
        return returns_df, volume_df

    
    def compute_rolling_covariance(self, returns_df: pd.DataFrame, 
                                 window: int = 60) -> pd.DataFrame:
        """
        Compute rolling covariance matrix.
        
        Args:
            returns_df: Daily returns DataFrame
            window: Rolling window size
            
        Returns:
            Rolling covariance matrix
        """
        logger.info(f"Computing rolling covariance with {window}-day window")
        
        # Simplified approach: compute rolling covariance directly
        # This avoids the complex MultiIndex issues
        rolling_cov = returns_df.rolling(window=window).cov()
        
        return rolling_cov
    
    def perform_pca_analysis(self, returns_df: pd.DataFrame, 
                           n_components: Optional[int] = None) -> Dict:
        """
        Perform Principal Component Analysis on returns with comprehensive interpretation.
        
        Args:
            returns_df: Daily returns DataFrame
            n_components: Number of components to extract
            
        Returns:
            Dictionary with PCA results and interpretation
        """
        logger.info("Performing PCA analysis")
        
        # Check if returns_df is empty or has insufficient data
        if returns_df.empty:
            logger.warning("Returns DataFrame is empty")
            return self._create_empty_pca_results()
        
        logger.info(f"Returns DataFrame shape: {returns_df.shape}")
        logger.info(f"Returns DataFrame columns: {returns_df.columns.tolist()}")
        
        # Check for sufficient data
        if len(returns_df) < 10:
            logger.warning(f"Insufficient data for PCA: {len(returns_df)} rows")
            return self._create_empty_pca_results()
        
        # Handle missing values more carefully
        returns_clean = returns_df.dropna(thresh=max(1, len(returns_df.columns) // 2))

        if len(returns_clean) < 10:
            logger.warning(f"After cleaning, insufficient data for PCA: {len(returns_clean)} rows")
            return self._create_empty_pca_results()
        
        logger.info(f"Clean returns DataFrame shape: {returns_clean.shape}")
        
        # Ensure we have at least 2 columns for PCA
        if len(returns_clean.columns) < 2:
            logger.warning(f"Insufficient columns for PCA: {len(returns_clean.columns)}")
            return self._create_empty_pca_results()
        
        # Standardize returns
        returns_scaled = self.scaler.fit_transform(returns_clean)
        
        # Perform PCA
        if n_components is None:
            n_components = min(len(returns_clean.columns), len(returns_clean), 5)  # Cap at 5 components
            
        if n_components < 1:
            n_components = 1
            
        logger.info(f"Performing PCA with {n_components} components on {returns_clean.shape} data")
        
        self.pca = PCA(n_components=n_components)
        pca_components = self.pca.fit_transform(returns_scaled)
        
        # Store results
        # pca.components_ has shape (n_components, n_features)
        # We want loadings with shape (n_features, n_components)
        self.factor_loadings = pd.DataFrame(
            self.pca.components_.T,  # Transpose to get (n_features, n_components)
            index=returns_clean.columns,  # Feature names as rows
            columns=[f'PC{i+1}' for i in range(n_components)]  # Component names as columns
        )
        
        self.explained_variance = pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Interpret principal components
        component_interpretations = self._interpret_principal_components()
        
        # Calculate factor importance scores
        factor_importance = self._calculate_factor_importance()
        
        # Create results dictionary with enhanced interpretation
        results = {
            'components': pca_components,
            'loadings': self.factor_loadings,
            'explained_variance': self.explained_variance,
            'cumulative_variance': self.explained_variance.cumsum(),
            'eigenvalues': self.pca.explained_variance_,
            'interpretations': component_interpretations,
            'factor_importance': factor_importance,
            'risk_decomposition': self._decompose_risk_factors(),
            'correlation_analysis': self._analyze_factor_correlations(),
            'data_info': {
                'original_shape': returns_df.shape,
                'clean_shape': returns_clean.shape,
                'n_components': n_components,
                'total_variance_explained': self.explained_variance.sum()
            }
        }
        
        logger.info(f"PCA complete. {n_components} components explain "
                   f"{self.explained_variance.sum():.1%} of variance")
        
        return results
    
    def _interpret_principal_components(self) -> Dict:
        """
        Interpret what each principal component represents.
        
        Returns:
            Dictionary with component interpretations
        """
        interpretations = {}
        
        for i, pc in enumerate(self.factor_loadings.columns):
            loadings = self.factor_loadings[pc].abs()
            top_assets = loadings.nlargest(3)
            
            # Determine component type based on loadings
            if 'SPY' in top_assets.index and top_assets['SPY'] > 0.5:
                component_type = "Market Risk Factor"
                description = "Represents broad market movements affecting all assets"
            elif 'USO' in top_assets.index and top_assets['USO'] > 0.5:
                component_type = "Oil Price Factor"
                description = "Captures direct oil price movements and energy commodity risk"
            elif 'XLE' in top_assets.index and top_assets['XLE'] > 0.5:
                component_type = "Energy Sector Factor"
                description = "Represents energy sector specific movements and sector rotation"
            elif 'XOP' in top_assets.index and top_assets['XOP'] > 0.5:
                component_type = "Oil & Gas Exploration Factor"
                description = "Captures exploration and production specific risks"
            elif 'BNO' in top_assets.index and top_assets['BNO'] > 0.5:
                component_type = "Brent Oil Factor"
                description = "Represents Brent crude oil specific movements"
            else:
                component_type = "Multi-Asset Factor"
                description = "Combined factor representing multiple asset correlations"
            
            # Calculate factor characteristics
            factor_volatility = self.explained_variance[pc]
            factor_contribution = self.explained_variance[pc] / self.explained_variance.sum()
            
            interpretations[pc] = {
                'type': component_type,
                'description': description,
                'top_assets': top_assets.to_dict(),
                'volatility_contribution': factor_volatility,
                'total_contribution': factor_contribution,
                'key_characteristics': self._get_factor_characteristics(pc)
            }
        
        return interpretations
    
    def _create_empty_pca_results(self) -> Dict:
        """
        Create empty PCA results when insufficient data is available.
        
        Returns:
            Dictionary with empty PCA results
        """
        logger.warning("Creating empty PCA results due to insufficient data")
        
        empty_df = pd.DataFrame()
        empty_series = pd.Series()
        
        return {
            'components': np.array([]),
            'loadings': empty_df,
            'explained_variance': empty_series,
            'cumulative_variance': empty_series,
            'eigenvalues': np.array([]),
            'interpretations': {},
            'factor_importance': empty_df,
            'risk_decomposition': {},
            'correlation_analysis': empty_df,
            'data_info': {
                'original_shape': (0, 0),
                'clean_shape': (0, 0),
                'n_components': 0,
                'total_variance_explained': 0.0,
                'error': 'Insufficient data for PCA analysis'
            }
        }
    
    def _get_factor_characteristics(self, pc: str) -> Dict:
        """
        Get detailed characteristics of a principal component.
        
        Args:
            pc: Principal component name
            
        Returns:
            Dictionary with factor characteristics
        """
        loadings = self.factor_loadings[pc]
        
        # Calculate factor characteristics
        characteristics = {
            'diversification_benefit': 1 - loadings.max(),  # Lower max loading = more diversification
            'sector_concentration': loadings.std(),  # Higher std = more concentrated
            'market_sensitivity': abs(loadings.get('SPY', 0)),  # Sensitivity to market
            'energy_sensitivity': abs(loadings.get('XLE', 0) + loadings.get('XOP', 0)),  # Energy sensitivity
            'commodity_sensitivity': abs(loadings.get('USO', 0) + loadings.get('BNO', 0))  # Commodity sensitivity
        }
        
        return characteristics
    
    def _calculate_factor_importance(self) -> pd.DataFrame:
        """
        Calculate importance scores for each factor.
        
        Returns:
            DataFrame with factor importance metrics
        """
        importance_data = []
        
        for pc in self.factor_loadings.columns:
            loadings = self.factor_loadings[pc]
            variance_explained = self.explained_variance[pc]
            
            # Calculate importance metrics
            importance_score = variance_explained * loadings.abs().mean()
            risk_contribution = variance_explained * loadings.abs().sum()
            diversification_score = 1 - loadings.abs().max()
            
            importance_data.append({
                'factor': pc,
                'variance_explained': variance_explained,
                'importance_score': importance_score,
                'risk_contribution': risk_contribution,
                'diversification_score': diversification_score,
                'max_loading': loadings.abs().max(),
                'avg_loading': loadings.abs().mean()
            })
        
        return pd.DataFrame(importance_data).set_index('factor')
    
    def _decompose_risk_factors(self) -> Dict:
        """
        Decompose total risk into factor contributions.
        
        Returns:
            Dictionary with risk decomposition
        """
        total_variance = self.explained_variance.sum()
        risk_decomposition = {}
        
        for pc in self.factor_loadings.columns:
            variance_contribution = self.explained_variance[pc]
            risk_decomposition[pc] = {
                'variance_contribution': variance_contribution,
                'percentage_contribution': variance_contribution / total_variance,
                'cumulative_contribution': self.explained_variance[:pc].sum() / total_variance
            }
        
        return risk_decomposition
    
    def _analyze_factor_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between principal components.
        
        Returns:
            DataFrame with factor correlations
        """
        # Get PCA components
        # Ensure we have the correct symbols that match the data
        if hasattr(self, 'factor_loadings') and not self.factor_loadings.empty:
            symbols = self.factor_loadings.index.tolist()
        else:
            symbols = self.symbols
            
        components_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(self.pca.components_))],
            index=symbols
        )
        
        # Calculate correlations
        correlations = components_df.T.corr()
        
        return correlations
    
    def compute_factor_exposures(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute XOP's exposure to principal components.
        
        Args:
            returns_df: Daily returns DataFrame
            
        Returns:
            DataFrame with factor exposures
        """
        if self.pca is None:
            raise ValueError("Must run PCA analysis first")
            
        logger.info("Computing factor exposures")
        
        # Get XOP returns
        xop_returns = returns_df['XOP']
        
        # Transform returns to PCA space
        returns_scaled = self.scaler.transform(returns_df)
        pca_components = self.pca.transform(returns_scaled)
        
        # Compute rolling factor exposures (30-day window)
        window = 30
        factor_exposures = pd.DataFrame(index=returns_df.index[window-1:])
        
        for i in range(window-1, len(returns_df)):
            # Use rolling window of returns
            window_returns = returns_scaled[i-window+1:i+1]
            window_pca = self.pca.transform(window_returns)
            
            # Regress XOP returns on PCA components
            xop_window = xop_returns.iloc[i-window+1:i+1].values
            exposures = np.linalg.lstsq(window_pca, xop_window, rcond=None)[0]
            
            for j, exposure in enumerate(exposures):
                factor_exposures.loc[returns_df.index[i], f'PC{j+1}_Exposure'] = exposure
        
        return factor_exposures
    
    def compute_mcvr(self, returns_df: pd.DataFrame, 
                    factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Marginal Contribution to Volatility Risk (MCVR).
        
        Args:
            returns_df: Daily returns DataFrame
            factor_exposures: Factor exposures DataFrame
            
        Returns:
            DataFrame with MCVR for each factor
        """
        logger.info("Computing Marginal Contribution to Volatility Risk")
        
        # Align data
        common_index = returns_df.index.intersection(factor_exposures.index)
        returns_aligned = returns_df.loc[common_index]
        exposures_aligned = factor_exposures.loc[common_index]
        
        # Compute rolling portfolio volatility
        window = 30
        mcvr_results = pd.DataFrame(index=common_index[window-1:])
        
        for i in range(window-1, len(common_index)):
            # Get window data
            window_returns = returns_aligned.iloc[i-window+1:i+1]
            window_exposures = exposures_aligned.iloc[i-window+1:i+1]
            
            # Compute covariance matrix
            cov_matrix = window_returns.cov()
            
            # Compute portfolio weights (assuming equal weight for simplicity)
            # Use actual columns from returns data, not self.symbols
            n_assets = len(window_returns.columns)
            weights = np.array([1/n_assets] * n_assets)
            
            # Compute portfolio volatility
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # Compute MCVR for each factor
            for factor in exposures_aligned.columns:
                if factor in window_exposures.columns:
                    factor_exposure = window_exposures[factor].mean()
                    # Simplified MCVR calculation
                    mcvr = factor_exposure * portfolio_vol
                    mcvr_results.loc[common_index[i], f'MCVR_{factor}'] = mcvr
        
        return mcvr_results
    
    def identify_market_regimes(self, returns_df: pd.DataFrame, 
                              vix_threshold: float = 30.0) -> pd.Series:
        """
        Identify market regimes based on volatility.
        
        Args:
            returns_df: Daily returns DataFrame
            vix_threshold: VIX threshold for bear market
            
        Returns:
            Series with market regime labels
        """
        logger.info("Identifying market regimes")
        
        # Fetch VIX data
        try:
            vix = yf.download('^VIX', start=returns_df.index[0], 
                            end=returns_df.index[-1])['Close']
            
            # Align with returns data
            vix_aligned = vix.reindex(returns_df.index).fillna(method='ffill')
            
            # Define regimes
            regimes = pd.Series(index=returns_df.index, dtype=str)
            regimes[vix_aligned > vix_threshold] = 'Bear'
            regimes[vix_aligned <= vix_threshold] = 'Bull'
            
            # Add sideways regime (low volatility)
            low_vol_threshold = 15.0
            regimes[vix_aligned <= low_vol_threshold] = 'Sideways'
            
        except Exception as e:
            logger.warning(f"Could not fetch VIX data: {e}")
            # Fallback: use SPY volatility
            spy_vol = returns_df['SPY'].rolling(30).std()
            regimes = pd.Series(index=returns_df.index, dtype=str)
            regimes[spy_vol > spy_vol.quantile(0.75)] = 'Bear'
            regimes[spy_vol < spy_vol.quantile(0.25)] = 'Sideways'
            regimes[regimes.isna()] = 'Bull'
        
        logger.info(f"Regime distribution: {regimes.value_counts().to_dict()}")
        return regimes
    
    def compute_volume_spike_features(self, volume_df, threshold=2.0, etfs=None):
        """
        Compute binary volume spike features for selected ETFs.
        Args:
            volume_df: DataFrame with daily volume
            threshold: Spike threshold (default 2.0)
            etfs: List of ETF symbols to compute spikes for
        Returns:
            DataFrame with binary spike features (1=spike, 0=no spike)
        """
        if etfs is None:
            etfs = ['XLE', 'USO', 'BNO']
        spikes = pd.DataFrame(index=volume_df.index)
        for etf in etfs:
            if etf in volume_df:
                avg20 = volume_df[etf].rolling(20).mean()
                spikes[etf + '_spike'] = (volume_df[etf] > threshold * avg20).astype(int)
        return spikes

    def run_complete_analysis(self, start_date: str, end_date: str, spike_threshold=2.0, spike_etfs=None) -> Dict:
        """
        Run complete factor exposure analysis, including volume spike features and macro data.
        """
        logger.info("Running complete factor exposure analysis")
        # Fetch data
        returns_df, volume_df = self.fetch_factor_data(start_date, end_date)
        returns_df.index = pd.to_datetime(returns_df.index)
        volume_df.index = pd.to_datetime(volume_df.index)
        # Fetch macro data (FRED: CPI, USD, rates)
        macro_data = {}
        if FRED_AVAILABLE:
            try:
                # CPI (Consumer Price Index)
                cpi = web.DataReader('CPIAUCSL', 'fred', returns_df.index[0], returns_df.index[-1])
                macro_data['cpi'] = cpi
                # USD Index (DXY)
                dxy = web.DataReader('DTWEXBGS', 'fred', returns_df.index[0], returns_df.index[-1])
                macro_data['usd'] = dxy
                # Fed Funds Rate
                rates = web.DataReader('FEDFUNDS', 'fred', returns_df.index[0], returns_df.index[-1])
                macro_data['rates'] = rates
            except Exception as e:
                logger.warning(f"Could not fetch FRED macro data: {e}")
        else:
            logger.warning("pandas_datareader not available, skipping FRED macro data.")
        # Perform PCA
        pca_results = self.perform_pca_analysis(returns_df)
        # Compute factor exposures
        factor_exposures = self.compute_factor_exposures(returns_df)
        # Compute MCVR
        mcvr_results = self.compute_mcvr(returns_df, factor_exposures)
        # Identify market regimes
        market_regimes = self.identify_market_regimes(returns_df)
        # Compute rolling covariance
        rolling_cov = self.compute_rolling_covariance(returns_df)
        # Compute volume spike features
        spike_features = self.compute_volume_spike_features(volume_df, threshold=spike_threshold, etfs=spike_etfs)
        results = {
            'returns_data': returns_df,
            'volume_data': volume_df,
            'pca_results': pca_results,
            'factor_exposures': factor_exposures,
            'mcvr_results': mcvr_results,
            'market_regimes': market_regimes,
            'rolling_covariance': rolling_cov,
            'volume_spike_features': spike_features,
            'macro_data': macro_data
        }
        logger.info("Factor exposure analysis complete")
        return results


def create_style_factors(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic style factors (momentum, volatility).
    
    Args:
        returns_df: Daily returns DataFrame
        
    Returns:
        DataFrame with style factors
    """
    logger.info("Creating style factors")
    
    factors = pd.DataFrame(index=returns_df.index)
    
    # Momentum factors (20-day, 60-day)
    for symbol in returns_df.columns:
        factors[f'{symbol}_momentum_20d'] = returns_df[symbol].rolling(20).mean()
        factors[f'{symbol}_momentum_60d'] = returns_df[symbol].rolling(60).mean()
    
    # Volatility factors
    for symbol in returns_df.columns:
        factors[f'{symbol}_volatility_20d'] = returns_df[symbol].rolling(20).std()
        factors[f'{symbol}_volatility_60d'] = returns_df[symbol].rolling(60).std()
    
    # Cross-sectional factors
    factors['momentum_spread'] = (returns_df['XOP'] - returns_df['SPY']).rolling(20).mean()
    factors['volatility_spread'] = (returns_df['XOP'].rolling(20).std() - 
                                   returns_df['SPY'].rolling(20).std())
    
    return factors 