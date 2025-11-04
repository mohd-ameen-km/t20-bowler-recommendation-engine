import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.utils.constants import BOWLING_TYPES

class MLBowlerRecommender:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        self.is_trained = False
        self.feature_columns = []
        self.bowling_types = BOWLING_TYPES

    def train_models(self, X, weakness_scores):
        """Train ML models and persist"""
        y = np.array([[weakness_scores[batter].get(bt, 0) for bt in self.bowling_types] for batter in X.index])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.kmeans.fit(X_train)
        self.model.fit(X_train, y_train)

        mse = mean_squared_error(y_test, self.model.predict(X_test))
        print(f"✅ Model trained. MSE: {mse:.3f}")
        self.is_trained = True

        joblib.dump(self.model, "models/rf_model.pkl")
        joblib.dump(self.kmeans, "models/kmeans.pkl")

    def load_models(self):
        try:
            self.model = joblib.load("models/rf_model.pkl")
            self.kmeans = joblib.load("models/kmeans.pkl")
            self.is_trained = True
            print("✅ Models loaded successfully.")
        except:
            print("⚠️ Models not found. Retraining required.")

    def predict_weakness(self, features):
        X_vec = np.array([[features.get(col, 0) for col in self.feature_columns]])
        preds = self.model.predict(X_vec)[0]
        return dict(zip(self.bowling_types, preds))

    def find_similar_batters(self, features, batters_data, top_n=5):
        X_vec = np.array([[features.get(col, 0) for col in self.feature_columns]])
        cluster = self.kmeans.predict(X_vec)[0]
        sims = [b for b, f in batters_data.items() if self.kmeans.predict(
            np.array([[f.get(col, 0) for col in self.feature_columns]])
        )[0] == cluster and b != features['batter_name']]
        return sims[:top_n]
