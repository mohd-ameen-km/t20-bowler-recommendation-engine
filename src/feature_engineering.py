import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []

    def prepare_features(self, batters_data):
        """Prepare feature matrix for ML"""
        df = pd.DataFrame(list(batters_data.values()))
        if len(df) == 0:
            return None, None

        df = df[df['total_balls'] >= 20]
        if len(df) == 0:
            return None, None

        # Select numeric columns
        X = df.select_dtypes(include=[np.number]).fillna(0)
        self.feature_columns = X.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, index=df['batter_name']), df['batter_name']

    def create_weakness_labels(self, batters_data, bowling_types):
        """Compute weakness scores per bowling type"""
        weakness = {}
        for batter, f in batters_data.items():
            scores = {}
            for bt in bowling_types:
                sr, dr, balls = f.get(f'{bt}_sr', 0), f.get(f'{bt}_dismissal_rate', 0), f.get(f'{bt}_balls_faced', 0)
                conf = min(balls / 15, 1.0)
                sr_norm = 1 - (sr / 200)
                dr_norm = dr / 100
                score = (2 * sr_norm * dr_norm / (sr_norm + dr_norm + 1e-6)) * conf * 100
                scores[bt] = score
            weakness[batter] = scores
        return weakness
