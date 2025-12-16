"""
ML-Based Stock Analyzer - Trained model for predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import pickle
import os
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")


class MLStockAnalyzer:
    """
    Machine Learning-based stock analyzer using trained models
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "random_forest"):
        """
        Initialize ML analyzer
        
        Args:
            model_path: Path to saved model file (if exists)
            model_type: Type of model ('random_forest', 'gradient_boosting')
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = model_path or "models/stock_predictor.pkl"
        self.feature_names = None
        
        # Load existing model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from technical indicators for ML model
        
        Args:
            df: DataFrame with price data and technical indicators
            
        Returns:
            DataFrame with feature columns
        """
        features_df = pd.DataFrame(index=df.index)
        
        # Price-based features
        if 'rsi' in df.columns:
            features_df['rsi'] = df['rsi']
            features_df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to -1 to 1
        else:
            features_df['rsi'] = 50
            features_df['rsi_normalized'] = 0
        
        # Moving average features
        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            features_df['ema_9_21_ratio'] = df['ema_9'] / (df['ema_21'] + 1e-10)
            features_df['price_ema9_ratio'] = df['close'] / (df['ema_9'] + 1e-10)
            features_df['price_ema21_ratio'] = df['close'] / (df['ema_21'] + 1e-10)
        else:
            features_df['ema_9_21_ratio'] = 1.0
            features_df['price_ema9_ratio'] = 1.0
            features_df['price_ema21_ratio'] = 1.0
        
        # MACD features
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            features_df['macd_diff'] = df['macd'] - df['macd_signal']
            features_df['macd_histogram'] = df.get('macd_histogram', 0)
        else:
            features_df['macd_diff'] = 0
            features_df['macd_histogram'] = 0
        
        # Bollinger Bands features
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_range = df['bb_upper'] - df['bb_lower']
            features_df['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-10)
            features_df['bb_width'] = bb_range / (df['close'] + 1e-10)
        else:
            features_df['bb_position'] = 0.5
            features_df['bb_width'] = 0
        
        # Price change features
        features_df['price_change_1d'] = df.get('change_1d', 0)
        features_df['price_change_5d'] = df.get('change_5d', 0)
        features_df['price_change_20d'] = df.get('change_20d', 0)
        
        # Volume features
        if 'volume' in df.columns and 'volume_ratio' in df.columns:
            features_df['volume_ratio'] = df['volume_ratio']
        else:
            features_df['volume_ratio'] = 1.0
        
        # ATR (volatility)
        if 'atr' in df.columns:
            features_df['atr_pct'] = (df['atr'] / (df['close'] + 1e-10)) * 100
        else:
            features_df['atr_pct'] = 0
        
        # Trend features
        if 'direction' in df.columns:
            features_df['trend_up'] = (df['direction'] == 'up').astype(int)
            features_df['trend_down'] = (df['direction'] == 'down').astype(int)
            features_df['trend_strength'] = df.get('strength', 0) / 100
        else:
            features_df['trend_up'] = 0
            features_df['trend_down'] = 0
            features_df['trend_strength'] = 0
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def prepare_target(self, df: pd.DataFrame, forward_periods: int = 5) -> pd.Series:
        """
        Prepare target variable (future returns)
        
        Args:
            df: DataFrame with price data
            forward_periods: Number of periods ahead to predict
            
        Returns:
            Series with future returns
        """
        # Calculate future return (forward_periods days ahead)
        future_price = df['close'].shift(-forward_periods)
        current_price = df['close']
        future_return = ((future_price - current_price) / (current_price + 1e-10)) * 100
        
        return future_return
    
    def train_model(self, historical_data: Dict[str, pd.DataFrame], 
                   forward_periods: int = 5, test_size: float = 0.2) -> Dict:
        """
        Train ML model on historical data
        
        Args:
            historical_data: Dict of ticker -> DataFrame with historical data
            forward_periods: Number of periods to predict ahead
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        print("🔄 Preparing training data...")
        
        # Collect features and targets from all stocks
        all_features = []
        all_targets = []
        
        for ticker, df in historical_data.items():
            if len(df) < 50:  # Need enough data
                continue
            
            # Calculate indicators if not present
            try:
                from core.analysis.technical_analyzer import calculate_all_indicators
                
                df_with_indicators = calculate_all_indicators(df)
                if len(df_with_indicators) < forward_periods + 20:
                    continue
                
                # Prepare features
                features = self.prepare_features(df_with_indicators)
                
                # Prepare target (future returns)
                target = self.prepare_target(df_with_indicators, forward_periods)
                
                # Remove rows with NaN targets (last forward_periods rows)
                valid_mask = ~target.isna()
                features = features[valid_mask]
                target = target[valid_mask]
                
                if len(features) > 0:
                    all_features.append(features)
                    all_targets.append(target)
            except Exception as e:
                print(f"   ⚠️  Skipping {ticker}: {str(e)[:100]}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data found")
        
        # Combine all data
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_targets, axis=0)
        
        # Remove any remaining NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"✅ Prepared {len(X)} training samples")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"🤖 Training {self.model_type} model...")
        
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'n_features': len(X.columns)
        }
        
        print(f"✅ Model trained!")
        print(f"   Train MAE: {train_mae:.2f}% | Test MAE: {test_mae:.2f}%")
        print(f"   Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
        print(f"   CV MAE: {cv_mae:.2f}%")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict future returns using trained model
        
        Args:
            df: DataFrame with current stock data and indicators
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare features
        features = self.prepare_features(df)
        
        # Get latest row
        latest_features = features.iloc[[-1]]
        
        # Ensure all expected features are present
        if self.feature_names:
            for col in self.feature_names:
                if col not in latest_features.columns:
                    latest_features[col] = 0
        
        # Reorder columns to match training
        if self.feature_names:
            latest_features = latest_features[self.feature_names]
        
        # Scale
        latest_scaled = self.scaler.transform(latest_features)
        
        # Predict
        predicted_return = self.model.predict(latest_scaled)[0]
        
        # Get feature importance for explanation
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = []
        
        # Convert to recommendation
        if predicted_return > 3:
            recommendation = "BUY"
            confidence = min(90, 60 + int(predicted_return))
        elif predicted_return > 1:
            recommendation = "CONSIDER BUY"
            confidence = min(75, 55 + int(predicted_return))
        elif predicted_return < -3:
            recommendation = "SELL"
            confidence = 70
        else:
            recommendation = "WAIT"
            confidence = 50
        
        reasoning = f"ML model predicts {predicted_return:.2f}% return"
        if top_features:
            reasoning += f". Top factors: {', '.join([f[0] for f in top_features])}"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'predicted_return': predicted_return,
            'reasoning': reasoning,
            'technical_score': min(100, max(0, 50 + predicted_return * 5)),
            'model_type': self.model_type,
            'upside_potential': 'High' if predicted_return > 5 else ('Medium' if predicted_return > 2 else 'Low'),
            'risk_level': 'Low' if predicted_return > 3 else ('Medium' if predicted_return > 0 else 'High')
        }
    
    def save_model(self, filepath: Optional[str] = None):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No model to save")
        
        filepath = filepath or self.model_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'trained_date': datetime.now().isoformat()
            }, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data.get('model_type', 'random_forest')
            self.feature_names = data.get('feature_names', None)
            self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")

