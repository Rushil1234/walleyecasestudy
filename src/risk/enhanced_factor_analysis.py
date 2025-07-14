"""
Enhanced Factor Analysis with Gradient-Boosted Trees and SHAP Analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedFactorAnalyzer:
    """
    Enhanced factor analysis using gradient-boosted trees with SHAP explainability.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize the enhanced factor analyzer.
        
        Args:
            symbols: List of symbols to analyze
        """
        self.symbols = symbols or ['XOP', 'XLE', 'USO', 'BNO', 'SPY', '^VIX', '^DXY', 'GC=F']
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.shap_explainer = None
        
    def fetch_factor_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch comprehensive factor data for analysis.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with factor data
        """
        logger.info(f"Fetching enhanced factor data for {self.symbols}")
        
        data = {}
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=start_date, end=end_date)
                # Ensure index is datetime and timezone-naive
                hist_data.index = pd.to_datetime(hist_data.index).tz_localize(None)
                
                # Calculate returns and technical indicators
                hist_data[f'{symbol}_returns'] = hist_data['Close'].pct_change()
                hist_data[f'{symbol}_volatility'] = hist_data[f'{symbol}_returns'].rolling(20).std()
                hist_data[f'{symbol}_momentum'] = hist_data['Close'].pct_change(20)
                hist_data[f'{symbol}_rsi'] = self._calculate_rsi(hist_data['Close'])
                
                data[symbol] = hist_data
                logger.info(f"Successfully fetched data for {symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        # Combine all data
        if not data:
            raise ValueError("No data could be fetched for any symbols")
            
        combined_data = pd.concat([df for df in data.values()], axis=1)
        combined_data = combined_data.fillna(method='ffill').dropna()
        
        return combined_data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features(self, data: pd.DataFrame, target_symbol: str = 'XOP') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for machine learning.
        
        Args:
            data: Raw factor data
            target_symbol: Symbol to predict
            
        Returns:
            Tuple of (features, target)
        """
        # Extract target (next day return)
        target = data[f'{target_symbol}_returns'].shift(-1).dropna()
        
        # Prepare features
        feature_columns = []
        for symbol in self.symbols:
            if symbol != target_symbol:
                feature_columns.extend([
                    f'{symbol}_returns',
                    f'{symbol}_volatility',
                    f'{symbol}_momentum',
                    f'{symbol}_rsi'
                ])
        
        # Add target symbol's own features (excluding returns)
        feature_columns.extend([
            f'{target_symbol}_volatility',
            f'{target_symbol}_momentum',
            f'{target_symbol}_rsi'
        ])
        
        # Filter to only include columns that exist in the data
        available_columns = [col for col in feature_columns if col in data.columns]
        features = data[available_columns].dropna()
        
        # Align features and target
        common_index = features.index.intersection(target.index)
        if len(common_index) == 0:
            raise ValueError("No common dates between features and target after alignment")
        
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        # Ensure we have enough data
        if len(features) < 50:
            raise ValueError(f"Insufficient data: only {len(features)} samples available, need at least 50")
        
        self.feature_names = available_columns
        
        logger.info(f"Prepared {len(features)} samples with {len(self.feature_names)} features")
        
        return features, target
    
    def train_gradient_boosted_model(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Train gradient-boosted tree model with time series cross-validation.
        
        Args:
            features: Feature matrix
            target: Target variable
            
        Returns:
            Training results dictionary
        """
        logger.info("Training gradient-boosted tree model...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8
        )
        
        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in tscv.split(features_scaled):
            X_train = features_scaled.iloc[train_idx]
            y_train = target.iloc[train_idx]
            X_val = features_scaled.iloc[val_idx]
            y_val = target.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            score = r2_score(y_val, y_pred)
            cv_scores.append(score)
        
        # Final training on full dataset
        self.model.fit(features_scaled, target)
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        results = {
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'model': self.model
        }
        
        logger.info(f"Model trained successfully. CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return results
    
    def analyze_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Perform comprehensive SHAP analysis.
        
        Args:
            features: Feature matrix
            target: Target variable
            
        Returns:
            SHAP analysis results
        """
        if self.model is None or self.shap_explainer is None:
            raise ValueError("Model must be trained before SHAP analysis")
        
        logger.info("Performing SHAP analysis...")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(features_scaled)
        
        # Feature importance summary
        feature_importance = dict(zip(self.feature_names, np.abs(shap_values).mean(0)))
        
        # SHAP summary plot data
        shap_summary = {
            'shap_values': shap_values,
            'feature_values': features_scaled.values,
            'feature_names': self.feature_names,
            'base_value': self.shap_explainer.expected_value
        }
        
        # Dependence plots for top features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        dependence_data = {}
        
        for feature_name, importance in top_features:
            feature_idx = self.feature_names.index(feature_name)
            dependence_data[feature_name] = {
                'shap_values': shap_values[:, feature_idx],
                'feature_values': features_scaled[feature_name].values,
                'importance': importance
            }
        
        results = {
            'feature_importance': feature_importance,
            'shap_summary': shap_summary,
            'dependence_data': dependence_data,
            'top_features': [f[0] for f in top_features]
        }
        
        return results
    
    def generate_strategy_insights(self, shap_results: Dict) -> Dict:
        """
        Generate actionable strategy insights from SHAP analysis.
        
        Args:
            shap_results: Results from SHAP analysis
            
        Returns:
            Strategy insights dictionary
        """
        insights = {
            'key_factors': [],
            'factor_interactions': [],
            'risk_indicators': [],
            'opportunity_signals': [],
            'recommendations': []
        }
        
        # Analyze top features
        top_features = shap_results['top_features'][:5]
        feature_importance = shap_results['feature_importance']
        
        for feature in top_features:
            importance = feature_importance[feature]
            
            # Categorize features
            if 'volatility' in feature:
                insights['risk_indicators'].append({
                    'factor': feature,
                    'importance': importance,
                    'description': f"High volatility in {feature.split('_')[0]} is a key risk indicator"
                })
            elif 'momentum' in feature:
                insights['opportunity_signals'].append({
                    'factor': feature,
                    'importance': importance,
                    'description': f"Momentum in {feature.split('_')[0]} provides strong directional signals"
                })
            elif 'rsi' in feature:
                insights['key_factors'].append({
                    'factor': feature,
                    'importance': importance,
                    'description': f"RSI in {feature.split('_')[0]} indicates overbought/oversold conditions"
                })
            else:
                insights['key_factors'].append({
                    'factor': feature,
                    'importance': importance,
                    'description': f"Returns in {feature.split('_')[0]} show strong predictive power"
                })
        
        # Generate recommendations
        if insights['risk_indicators']:
            insights['recommendations'].append(
                "Monitor volatility indicators closely - they are key risk predictors"
            )
        
        if insights['opportunity_signals']:
            insights['recommendations'].append(
                "Momentum factors provide strong directional signals for entry/exit"
            )
        
        if insights['key_factors']:
            insights['recommendations'].append(
                "Focus on RSI and return-based factors for optimal timing"
            )
        
        return insights
    
    def create_visualizations(self, shap_results: Dict, save_path: str = None) -> Dict:
        """
        Create comprehensive visualizations of SHAP analysis.
        
        Args:
            shap_results: Results from SHAP analysis
            save_path: Path to save plots
            
        Returns:
            Dictionary with plot paths
        """
        plots = {}
        
        # SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_results['shap_summary']['shap_values'],
            shap_results['shap_summary']['feature_values'],
            feature_names=shap_results['shap_summary']['feature_names'],
            show=False
        )
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()
        
        if save_path:
            summary_path = f"{save_path}_shap_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plots['shap_summary'] = summary_path
        plt.close()
        
        # Feature Importance Bar Plot
        plt.figure(figsize=(12, 8))
        importance_df = pd.DataFrame(
            list(shap_results['feature_importance'].items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['Importance'])
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('SHAP Importance')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        
        if save_path:
            importance_path = f"{save_path}_feature_importance.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plots['feature_importance'] = importance_path
        plt.close()
        
        # Dependence plots for top features
        for feature_name, data in shap_results['dependence_data'].items():
            plt.figure(figsize=(10, 6))
            plt.scatter(data['feature_values'], data['shap_values'], alpha=0.6)
            plt.xlabel(feature_name)
            plt.ylabel('SHAP Value')
            plt.title(f'SHAP Dependence Plot: {feature_name}')
            plt.tight_layout()
            
            if save_path:
                dep_path = f"{save_path}_dependence_{feature_name.replace('_', '')}.png"
                plt.savefig(dep_path, dpi=300, bbox_inches='tight')
                plots[f'dependence_{feature_name}'] = dep_path
            plt.close()
        
        return plots
    
    def run_complete_analysis(self, start_date: str, end_date: str, 
                            target_symbol: str = 'XOP') -> Dict:
        """
        Run complete enhanced factor analysis.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            target_symbol: Symbol to predict
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting enhanced factor analysis for {target_symbol}")
        
        # Fetch data
        data = self.fetch_factor_data(start_date, end_date)
        
        # Prepare features
        features, target = self.prepare_features(data, target_symbol)
        
        # Train model
        training_results = self.train_gradient_boosted_model(features, target)
        
        # SHAP analysis
        shap_results = self.analyze_feature_importance(features, target)
        
        # Generate insights
        insights = self.generate_strategy_insights(shap_results)
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots = self.create_visualizations(shap_results, f"data/plots/enhanced_factor_analysis_{timestamp}")
        
        results = {
            'training_results': training_results,
            'shap_results': shap_results,
            'insights': insights,
            'plots': plots,
            'data_info': {
                'start_date': start_date,
                'end_date': end_date,
                'target_symbol': target_symbol,
                'n_samples': len(features),
                'n_features': len(self.feature_names)
            }
        }
        
        logger.info("Enhanced factor analysis completed successfully")
        
        return results 