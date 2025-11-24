import pandas as pd
import os

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        """Loads data from CSV file."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        try:
            df = pd.read_csv(self.filepath)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def validate_data(self, df):
        """Validates if required columns exist."""
        required_columns = [
            'Study Hours', 'Attendance (%)', 'Previous Scores', 
            'Assignment Completion', 'Sleep Hours', 'Social Media Usage', 
            'Performance Index'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        return True
