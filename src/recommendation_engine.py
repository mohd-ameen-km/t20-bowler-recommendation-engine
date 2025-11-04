import numpy as np
import pandas as pd
from src.ml_models import MLBowlerRecommender
from src.feature_engineering import FeatureEngineer

class EnhancedBowlerRecommender:
    def __init__(self, data_processor, min_balls_threshold=5):
        """
        Main engine for recommending optimal bowling types against a batter.
        Uses ML if trained, otherwise falls back to statistical heuristics.
        """
        self.dp = data_processor
        self.fe = FeatureEngineer()
        self.ml = MLBowlerRecommender()
        self.batters_data = {}
        self.is_trained = False
        self.min_balls_threshold = min_balls_threshold

    # ===========================
    # ⚙️ TRAINING PIPELINE
    # ===========================
    def prepare_ml_data(self):
        """Prepares batter-level data and trains ML models."""
        all_batters = self.dp.valid_balls['bat'].unique()

        for batter in all_batters:
            f = self.dp.get_batter_features(batter, min_balls_threshold=self.min_balls_threshold)
            if f:
                self.batters_data[batter] = f

        X, names = self.fe.prepare_features(self.batters_data)
        if X is None:
            print("❌ Not enough data to train ML models.")
            return False

        # Dynamically get unique bowling types from dataset
        all_bowling_types = self.dp.valid_balls['bowling_type'].unique().tolist()

        weakness_labels = self.fe.create_weakness_labels(self.batters_data, all_bowling_types)
        self.ml.feature_columns = self.fe.feature_columns
        self.ml.train_models(X, weakness_labels)
        self.is_trained = True
        print("✅ ML models trained successfully.")
        return True

    # ===========================
    # 🎯 RECOMMENDATION LOGIC
    # ===========================
    def recommend(self, batter_name, phase=None):
        """
        Generates bowling-type recommendation for a given batter and phase.
        Uses trained ML model if available; otherwise uses statistical fallback.
        """
        f = self.dp.get_batter_features(batter_name, phase, min_balls_threshold=self.min_balls_threshold)
        if not f:
            return {
                'method': 'No Data',
                'recommended_type': None,
                'weakness_score': None,
                'all_predictions': {},
                'similar_batters': []
            }

        if self.is_trained:
            return self._ml_recommendation(f)
        else:
            return self._statistical_recommendation(f)

    # ===========================
    # 🧠 ML-BASED RECOMMENDATION
    # ===========================
    def _ml_recommendation(self, batter_features):
        """
        Predicts weakness scores for all bowling types using trained ML model.
        """
        predictions = self.ml.predict_weakness(batter_features)

        # Filter out types the batter never faced or faced very few balls
        valid_predictions = {}
        for bt, score in predictions.items():
            balls = batter_features.get(f"{bt}_balls_faced", 0)
            if balls >= self.min_balls_threshold:
                valid_predictions[bt] = score

        if not valid_predictions:
            return {
                'method': 'ML',
                'recommended_type': 'No Reliable Data',
                'weakness_score': 0,
                'all_predictions': {},
                'similar_batters': []
            }

        # Pick max weakness (higher = weaker against)
        best_type = max(valid_predictions.items(), key=lambda x: x[1])
        similar_batters = self.ml.find_similar_batters(batter_features, self.batters_data)

        return {
            'method': 'ML',
            'recommended_type': best_type[0],
            'weakness_score': best_type[1],
            'all_predictions': valid_predictions,
            'similar_batters': similar_batters
        }

    # ===========================
    # 📊 STATISTICAL RECOMMENDATION
    # ===========================
    def _statistical_recommendation(self, batter_features):
        """
        Calculates weakness using strike rate, dismissal rate, and confidence.
        Skips bowling types where batter faced too few deliveries.
        """
        weakness_scores = {}

        for key in batter_features.keys():
            if key.endswith('_sr'):
                base = key[:-3]  # strip '_sr' suffix
                sr = batter_features.get(f"{base}_sr", None)
                dr = batter_features.get(f"{base}_dismissal_rate", None)
                balls = batter_features.get(f"{base}_balls_faced", 0)

                if sr is None or dr is None or balls < self.min_balls_threshold:
                    continue

                confidence = min(balls / 10, 1.0)
                weakness_score = ((100 - sr) * 0.6 + dr * 0.4) * confidence
                weakness_scores[base] = weakness_score

        if not weakness_scores:
            return {
                'method': 'Statistical',
                'recommended_type': 'No Reliable Data',
                'weakness_score': 0,
                'all_predictions': {},
                'similar_batters': []
            }

        best_type = max(weakness_scores.items(), key=lambda x: x[1])

        return {
            'method': 'Statistical',
            'recommended_type': best_type[0],
            'weakness_score': best_type[1],
            'all_predictions': weakness_scores,
            'similar_batters': []
        }
