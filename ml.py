import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys
import argparse
from db import DatabaseManager, Trade, Signal
from logging_config import get_logger
from datetime import datetime


logger = get_logger(__name__, structured_format=True)


class MLFilter:
    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volume_ratio', 'trend_score', 'volatility',
            'price_change_1h', 'price_change_4h', 'price_change_24h'
        ]
        self.model_path = "ml_model.joblib"
        self.scaler_path = "ml_scaler.joblib"
        self.last_accuracy: float | None = None
        self.last_trained: str | None = None
        self.load_model()

    def prepare_features(self, indicators: Dict[str, float]) -> np.ndarray:
        """Convert indicator dict to feature array"""
        try:
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_histogram = indicators.get('macd_histogram', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            trend_score = indicators.get('trend_score', 0)
            volatility = indicators.get('volatility', 0)

            price = indicators.get('price', 0)
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            bb_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5

            price_change_1h = indicators.get('price_change_1h', 0)
            price_change_4h = indicators.get('price_change_4h', 0)
            price_change_24h = indicators.get('price_change_24h', 0)

            return np.array([
                rsi, macd, macd_signal, macd_histogram,
                bb_position, volume_ratio, trend_score, volatility,
                price_change_1h, price_change_4h, price_change_24h
            ]).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.zeros((1, len(self.feature_columns)))

    def train_model(self, trades: List[Trade]) -> bool:
        """Train ML model from database trades"""
        try:
            if len(trades) < 10:
                logger.warning("Few trades available, training may be unreliable")

            X, y = [], []
            for trade in trades:
                indicators = getattr(trade, 'indicators', {})
                if not indicators:
                    continue
                features = self.prepare_features(indicators)
                X.append(features.flatten())
                y.append(1 if (trade.pnl or 0) > 0 else 0)

            if len(X) < 5:
                logger.error("Not enough valid training samples")
                return False

            X = np.array(X)
            y = np.array(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
            )
            self.model.fit(X_train_scaled, y_train)

            y_pred = self.model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            self.last_accuracy = float(acc)
            self.last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"ML model trained with accuracy: {acc:.3f}")
            logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

            self.save_model()
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict_signal_quality(self, indicators: Dict[str, float]) -> float:
        try:
            if not self.model:
                return 0.5
            features = self.prepare_features(indicators)
            features_scaled = self.scaler.transform(features)
            return float(self.model.predict_proba(features_scaled)[0][1])
        except Exception as e:
            logger.error(f"Error predicting signal quality: {e}")
            return 0.5
    
    def filter_signals(self, signals: List[Any], threshold: float = 0.4) -> List[Any]:
        """Filter signals based on ML model prediction"""
        filtered_signals: List[Any] = []
        for sig in signals:
            # handle both Signal objects and dicts
            if hasattr(sig, "indicators"):
                indicators = sig.indicators or {}
                score = getattr(sig, "score", 0) or 0
            elif isinstance(sig, dict):
                indicators = sig.get("indicators", {})
                score = sig.get("score", 0) or 0
            else:
                continue

            quality = self.predict_signal_quality(indicators)
            if quality >= threshold:
                new_score = min(100, score * (1 + quality))
                if hasattr(sig, "score"):
                    setattr(sig, "score", new_score)
                elif isinstance(sig, dict):
                    sig["score"] = new_score
                filtered_signals.append(sig)

        logger.info(f"{len(filtered_signals)}/{len(signals)} signals passed ML filter")
        return filtered_signals


    def get_feature_importance(self) -> Dict[str, float] | None:
        """Return feature importance if model is trained"""
        if self.model:
            importance = self.model.feature_importances_
            return {name: float(imp) for name, imp in zip(self.feature_columns, importance)}
        return None

    def get_status(self) -> Dict[str, Any]:
        """Return status info about the ML model"""
        try:
            return {
                "model_loaded": self.model is not None,
                "model_path": self.model_path if self.model else None,
                "scaler_path": self.scaler_path if self.scaler else None,
                "feature_count": len(self.feature_columns),
                "features": self.feature_columns,
                "mode": "Trained" if self.model else "Untrained",
                "last_accuracy": round(self.last_accuracy, 3) if self.last_accuracy else None,
                "last_trained": self.last_trained
            }
        except Exception as e:
            logger.error(f"Error getting ML status: {e}")
            return {"model_loaded": False, "error": str(e)}


    def save_model(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("ML model saved")

    def load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("ML model loaded")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML Signal Filter")
    parser.add_argument('--train', action='store_true', help="Train model from DB trades")
    parser.add_argument('--threshold', type=float, default=0.6, help="ML score threshold")
    args = parser.parse_args()

    db_manager = DatabaseManager()
    ml_filter = MLFilter()

    try:
        if args.train:
            trades = db_manager.get_trades(limit=1000)
            if not trades:
                logger.error("No trades found to train model")
                sys.exit(1)
            success = ml_filter.train_model(trades)
            print({"success": success})
        else:
            signals = db_manager.get_signals(limit=100)
            if not signals:
                logger.warning("No signals found")
                print([])
                sys.exit(0)
            filtered = ml_filter.filter_signals(signals, threshold=args.threshold)
            print([s.to_dict() for s in filtered])
    finally:
        db_manager.close()
