from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def clean_data(self, df):
        """Handles missing values."""
        # For simplicity, dropping rows with missing values
        # In a real scenario, imputation might be better
        return df.dropna()

    def split_data(self, df, target_col='Performance Index', test_size=0.2):
        """Splits data into train and test sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def preprocess_features(self, X_train, X_test):
        """Scales numerical features."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for better readability if needed, 
        # but sklearn models handle arrays fine.
        return X_train_scaled, X_test_scaled
