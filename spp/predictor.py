import joblib
import os
import numpy as np

class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Loads the model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)

    def predict(self, input_data):
        """Predicts performance for new data."""
        if not self.model:
            self.load_model()
        
        # Ensure input is 2D array
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
            
        prediction = self.model.predict(input_data)
        return prediction[0]
