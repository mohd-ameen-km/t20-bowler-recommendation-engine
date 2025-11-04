import numpy as np
from src.ml_models import MLBowlerRecommender
from src.feature_engineering import FeatureEngineer

class EnhancedBowlerRecommender:
    def __init__(self, data_processor, min_balls_threshold=5):
        self.dp = data_processor
        self.fe = FeatureEngineer()
        self.ml = MLBowlerRecommender()
        self.batters_data = {}
        self.is_trained = False
        self.min_balls_threshold = min_balls_threshold

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

        all_bowling_types = self.dp.valid_balls['bowling_type'].unique().tolist()
        weakness_labels = self.fe.create_weakness_labels(self.batters_data, all_bowling_types)
        self.ml.feature_columns = self.fe.feature_columns
        self.ml.train_models(X, weakness_labels)
        self.is_trained = True
        return True

    def recommend(self, batter_name, phase=None):
        f = self.dp.get_batter_features(batter_name, phase, min_balls_threshold=self.min_balls_threshold)
        if not f:
            return {
                'method': 'No Data',
                'recommended_type': None,
                'weakness_score': None,
                'all_predictions': {},
                'similar_batters': []
            }

        if self.is_trained and self.ml.is_trained:
            return self._ml_recommendation(f)
        else:
            return self._statistical_recommendation(f)

    def _ml_recommendation(self, batter_features):
        predictions = self.ml.predict_weakness(batter_features)
        valid_predictions = {bt: s for bt, s in predictions.items()
                             if batter_features.get(f"{bt}_balls_faced", 0) >= self.min_balls_threshold}
        if not valid_predictions:
            return {'method': 'ML', 'recommended_type': 'No Reliable Data', 'weakness_score': 0,
                    'all_predictions': {}, 'similar_batters': []}
        best_type = max(valid_predictions.items(), key=lambda x: x[1])
        similar = self.ml.find_similar_batters(batter_features, self.batters_data)
        return {'method': 'ML', 'recommended_type': best_type[0], 'weakness_score': best_type[1],
                'all_predictions': valid_predictions, 'similar_batters': similar}

    def _statistical_recommendation(self, batter_features):
        weakness_scores = {}
        for key in batter_features.keys():
            if key.endswith('_sr'):
                base = key[:-3]
                sr = batter_features.get(f"{base}_sr", None)
                dr = batter_features.get(f"{base}_dismissal_rate", None)
                balls = batter_features.get(f"{base}_balls_faced", 0)
                if sr is None or dr is None or balls < self.min_balls_threshold:
                    continue
                confidence = min(balls / 10, 1.0)
                weakness_score = ((100 - sr) * 0.6 + dr * 0.4) * confidence
                weakness_scores[base] = weakness_score
        if not weakness_scores:
            return {'method': 'Statistical', 'recommended_type': 'No Reliable Data', 'weakness_score': 0,
                    'all_predictions': {}, 'similar_batters': []}
        best_type = max(weakness_scores.items(), key=lambda x: x[1])
        return {'method': 'Statistical', 'recommended_type': best_type[0],
                'weakness_score': best_type[1], 'all_predictions': weakness_scores, 'similar_batters': []}
