import pandas as pd
import numpy as np

class T20DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Clean and preprocess T20 dataset."""
        # Handle missing and case formatting
        self.df['bowl_style'] = self.df['bowl_style'].astype(str).str.strip().str.upper()
        self.df['bowl_kind'] = self.df['bowl_kind'].astype(str).str.strip().str.upper()
        self.df['bat'] = self.df['bat'].astype(str).str.strip()
        self.df['bowl'] = self.df['bowl'].astype(str).str.strip()
        
        # Over & Phase calculation
        self.df['over'] = self.df['ball'].astype(int)
        self.df['phase'] = self.df['over'].apply(self._get_match_phase)
        
        # Filter valid balls only
        self.valid_balls = self.df[(self.df['wide'] == 0) & (self.df['noball'] == 0)]
        
        # Define the "real" bowling type column (use actual dataset info)
        self.valid_balls['bowling_type'] = self.valid_balls.apply(self._define_real_bowling_type, axis=1)
    
    def _get_match_phase(self, over):
        """Classify overs into phases."""
        if over <= 6:
            return 'Powerplay'
        elif 7 <= over <= 11:
            return 'Middle1'
        elif 12 <= over <= 16:
            return 'Middle2'
        else:
            return 'Death'
    
    def _define_real_bowling_type(self, row):
        """Combine style and kind to represent real bowling type."""
        style = row['bowl_style']
        kind = row['bowl_kind']
        if style and style != 'NAN':
            if kind and kind != 'NAN':
                return f"{style} - {kind}".strip()
            return style.strip()
        elif kind and kind != 'NAN':
            return kind.strip()
        return 'UNKNOWN'

    def get_batter_features(self, batter_name, phase=None, min_balls_threshold=5):
        """Extract batting performance features vs actual bowling types."""
        df_batter = self.valid_balls[self.valid_balls['bat'] == batter_name]
        if phase:
            df_batter = df_batter[df_batter['phase'] == phase]
        if len(df_batter) == 0:
            return None

        features = {
            'batter_name': batter_name,
            'total_balls': len(df_batter),
            'total_runs': df_batter['batruns'].sum(),
            'strike_rate': (df_batter['batruns'].sum() / len(df_batter)) * 100,
            'dismissals': df_batter['out'].sum(),
            'boundary_percentage': (len(df_batter[df_batter['batruns'].isin([4, 6])]) / len(df_batter)) * 100
        }

        # Group by real bowling type
        grouped = df_batter.groupby('bowling_type').agg({
            'batruns': ['sum', 'count'],
            'out': 'sum'
        })

        # Compute per-type metrics, skipping sparse data
        for bowl_type in grouped.index:
            runs = grouped.loc[bowl_type, ('batruns', 'sum')]
            balls = grouped.loc[bowl_type, ('batruns', 'count')]
            outs = grouped.loc[bowl_type, ('out', 'sum')]
            if balls < min_balls_threshold:
                # Skip sparse categories (don’t include them in weakness calc)
                continue

            sr = (runs / balls) * 100
            dr = (outs / balls) * 100
            features[f'{bowl_type}_sr'] = sr
            features[f'{bowl_type}_dismissal_rate'] = dr
            features[f'{bowl_type}_balls_faced'] = balls
        
        return features
