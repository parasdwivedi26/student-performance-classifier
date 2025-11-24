from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.model = None

    def train_model(self, X_train, y_train, model_type='rf'):
        """Trains the specified model."""
        if model_type == 'lr':
            self.model = LinearRegression()
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model type. Choose 'lr' or 'rf'.")
        
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluates the model and returns metrics."""
        if not self.model:
            raise Exception("Model not trained yet.")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        return metrics, y_pred

    def save_model(self, filepath):
        """Saves the trained model to a file."""
        if not self.model:
            raise Exception("Model not trained yet.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
