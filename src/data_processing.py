import pandas as pd
import numpy as np

class T20DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self._preprocess_data()

    # ===============================
    # 🔧 DATA CLEANING AND PHASE SETUP
    # ===============================
    def _preprocess_data(self):
        """Clean and preprocess T20 dataset."""
        self.df['bowl_style'] = self.df['bowl_style'].astype(str).str.strip().str.upper()
        self.df['bowl_kind'] = self.df['bowl_kind'].astype(str).str.strip().str.upper()
        self.df['bat'] = self.df['bat'].astype(str).str.strip()
        self.df['bowl'] = self.df['bowl'].astype(str).str.strip()

        # Handle missing and invalid numeric values
        self.df['wide'] = pd.to_numeric(self.df['wide'], errors='coerce').fillna(0).astype(int)
        self.df['noball'] = pd.to_numeric(self.df['noball'], errors='coerce').fillna(0).astype(int)
        self.df['over'] = pd.to_numeric(self.df['over'], errors='coerce').fillna(0).astype(int)

        # Determine match phase based on over number
        self.df['phase'] = self.df['over'].apply(self._get_match_phase)

        # Filter valid deliveries (excluding wides/noballs)
        self.valid_balls = self.df[(self.df['wide'] == 0) & (self.df['noball'] == 0)].copy()

        # Define a realistic bowling type field
        self.valid_balls['bowling_type'] = self.valid_balls.apply(self._define_real_bowling_type, axis=1)

    def _get_match_phase(self, over):
        """Classify overs into standard T20 phases."""
        if over <= 6:
            return 'Powerplay'
        elif 7 <= over <= 11:
            return 'Middle1'
        elif 12 <= over <= 16:
            return 'Middle2'
        elif over >= 17:
            return 'Death'
        return 'Unknown'

    def _define_real_bowling_type(self, row):
        """Derive actual bowling type from style and kind."""
        style, kind = row['bowl_style'], row['bowl_kind']
        if style and style != 'NAN':
            if kind and kind != 'NAN':
                return f"{style} - {kind}".strip()
            return style.strip()
        elif kind and kind != 'NAN':
            return kind.strip()
        return 'UNKNOWN'

    # ======================================
    # ⚙️ BATTER-WISE FEATURE EXTRACTION
    # ======================================
    def get_batter_features(self, batter_name, phase=None, min_balls_threshold=5):
        """Extract performance stats vs each bowling type."""
        df_batter = self.valid_balls[self.valid_balls['bat'] == batter_name]
        if phase:
            df_batter = df_batter[df_batter['phase'] == phase]

        if len(df_batter) == 0:
            return None

        features = {
            'batter_name': batter_name,
            'total_balls': len(df_batter),
            'total_runs': df_batter['batruns'].sum(),
            'strike_rate': (df_batter['batruns'].sum() / len(df_batter)) * 100 if len(df_batter) else 0,
            'dismissals': df_batter['out'].astype(int).sum(),
            'boundary_percentage': (len(df_batter[df_batter['batruns'].isin([4, 6])]) / len(df_batter)) * 100
        }

        grouped = df_batter.groupby('bowling_type').agg({
            'batruns': ['sum', 'count'],
            'out': 'sum'
        })

        for bowl_type in grouped.index:
            runs = grouped.loc[bowl_type, ('batruns', 'sum')]
            balls = grouped.loc[bowl_type, ('batruns', 'count')]
            outs = grouped.loc[bowl_type, ('out', 'sum')]
            if balls < min_balls_threshold:
                continue
            sr = (runs / balls) * 100
            dr = (outs / balls) * 100
            features[f'{bowl_type}_sr'] = sr
            features[f'{bowl_type}_dismissal_rate'] = dr
            features[f'{bowl_type}_balls_faced'] = balls

        # Ensure consistent bowling type keys
        all_types = self.valid_balls['bowling_type'].unique()
        for bt in all_types:
            features.setdefault(f'{bt}_sr', 0)
            features.setdefault(f'{bt}_dismissal_rate', 0)
            features.setdefault(f'{bt}_balls_faced', 0)

        return features
