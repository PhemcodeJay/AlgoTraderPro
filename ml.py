import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json
import sys
import argparse
from db import DatabaseManager
from numpy.typing import NDArray

# Configure logging to match db.py
from logging_config import get_logger
logger = get_logger(__name__, structured_format=True)

class MLFilter:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volume_ratio', 'trend_score', 'volatility',
            'price_change_1h', 'price_change_4h', 'price_change_24h'
        ]
        self.model_path = "ml_model.joblib"
        self.scaler_path = "ml_scaler.joblib"
        
        # Load existing model if available
        self.load_model()

    def prepare_features(self, indicators: Dict[str, Any]) -> NDArray[np.float64]:
        """Prepare features from indicators for ML prediction"""
        try:
            # Extract basic indicators
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_histogram = indicators.get('macd_histogram', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            trend_score = indicators.get('trend_score', 0)
            volatility = indicators.get('volatility', 0)
            
            # Calculate Bollinger Band position
            price = indicators.get('price', 0)
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            bb_middle = indicators.get('bb_middle', 0)
            
            if bb_upper > bb_lower:
                bb_position = (price - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5
            
            # Price change features
            price_change_1h = indicators.get('price_change_1h', 0)
            price_change_4h = indicators.get('price_change_4h', 0)
            price_change_24h = indicators.get('price_change_24h', 0)
            
            features = np.array([
                rsi, macd, macd_signal, macd_histogram,
                bb_position, volume_ratio, trend_score, volatility,
                price_change_1h, price_change_4h, price_change_24h
            ]).reshape(1, -1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([]).reshape(1, -1)

    def train_model(self, training_data: List[Dict]) -> bool:
        """Train the ML model with historical trade data"""
        try:
            if len(training_data) < 100:
                logger.warning("Insufficient training data")
                return False
            
            # Prepare training features and labels
            X = []
            y = []
            
            for data in training_data:
                features = self.prepare_features(data['indicators'])
                if features.shape[1] > 0:
                    X.append(features.flatten())
                    # Label: 1 for profitable (pnl > 0), 0 for unprofitable
                    y.append(1 if data.get('profit', 0) > 0 else 0)
            
            if len(X) < 50:
                logger.error("Not enough valid training samples")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            # Save model
            self.save_model()
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict_signal_quality(self, indicators: Dict[str, Any]) -> float:
        """Predict signal quality (0-1 probability of success)"""
        try:
            if self.model is None:
                logger.warning("No trained model available")
                return 0.5
            
            features = self.prepare_features(indicators)
            if features.shape[1] == 0:
                return 0.5
            
            features_scaled = self.scaler.transform(features)
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error predicting signal quality: {e}")
            return 0.5

    def filter_signals(self, signals: List[Dict], threshold: float = 0.6) -> List[Dict]:
        """Filter signals using ML model predictions"""
        try:
            if self.model is None:
                logger.warning("No ML model available, returning all signals")
                return signals
            
            filtered_signals = []
            
            for signal in signals:
                indicators = signal.get('indicators', {})
                quality_score = self.predict_signal_quality(indicators)
                
                # Add ML score to signal
                signal['ml_score'] = quality_score
                
                # Filter based on threshold
                if quality_score >= threshold:
                    # Adjust signal score based on ML prediction
                    original_score = signal.get('score', 0)
                    enhanced_score = min(100, original_score * (1 + quality_score))
                    signal['score'] = enhanced_score
                    filtered_signals.append(signal)
            
            logger.info(f"ML filter: {len(filtered_signals)}/{len(signals)} signals passed")
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return signals

    def save_model(self):
        """Save trained model and scaler"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                logger.info("ML model saved")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self):
        """Load existing model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("ML model loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def update_model_with_feedback(self, signal: Dict, outcome: bool):
        """Update model with feedback from trade outcome"""
        try:
            logger.info(f"Feedback received for {signal.get('symbol')}: {outcome}")
            
            feedback_file = "ml_feedback.json"
            feedback_data = {
                'signal': signal,
                'outcome': outcome,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            feedbacks = []
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    feedbacks = json.load(f)
            
            feedbacks.append(feedback_data)
            feedbacks = feedbacks[-1000:]
            
            with open(feedback_file, 'w') as f:
                json.dump(feedbacks, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating model with feedback: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if self.model is None:
                return {}
            
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))
            
            return dict(sorted(feature_importance.items(), key=lambda x: float(x[1]), reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML Signal Filter")
    parser.add_argument('--train', action='store_true', help="Train the model with trade data from database")
    parser.add_argument('--threshold', type=float, default=0.6, help="ML score threshold for filtering")
    args = parser.parse_args()

    # Initialize database manager
    db_manager = DatabaseManager()
    ml_filter = MLFilter()

    try:
        if args.train:
            # Fetch trades from database for training
            trades = db_manager.get_trades(limit=1000)
            if not trades:
                logger.error("No trades found in database for training")
                print(json.dumps({'success': False, 'error': 'No trades found'}))
                sys.exit(1)
            
            # Map trades to training data format expected by train_model
            training_data = [
                {
                    'indicators': trade.to_dict().get('indicators', {}),
                    'profit': trade.pnl if trade.pnl is not None else 0
                }
                for trade in trades
                if 'indicators' in trade.to_dict() and trade.pnl is not None
            ]
            
            success = ml_filter.train_model(training_data)
            print(json.dumps({'success': success}))
        else:
            # Fetch signals from database
            signals = [signal.to_dict() for signal in db_manager.get_signals(limit=100)]
            if not signals:
                logger.error("No signals found in database")
                print(json.dumps([]))
                sys.exit(0)
            
            # Filter signals
            filtered_signals = ml_filter.filter_signals(signals, threshold=args.threshold)
            # Output as JSON
            print(json.dumps(filtered_signals, indent=2))
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(json.dumps({'error': str(e)}))
        sys.exit(1)
    
    finally:
        db_manager.close()