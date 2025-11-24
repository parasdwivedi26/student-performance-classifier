import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spp.data_loader import DataLoader
from spp.preprocessor import Preprocessor
from spp.trainer import ModelTrainer
from spp.predictor import Predictor
from spp.visualizer import Visualizer

def verify_pipeline():
    print("Starting pipeline verification...")
    
    # 1. Data Loading
    data_path = 'SPP/data/student_data.csv'
    loader = DataLoader(data_path)
    df = loader.load_data()
    loader.validate_data(df)
    print("[PASS] Data loaded and validated.")
    
    # 2. Preprocessing
    preprocessor = Preprocessor()
    df = preprocessor.clean_data(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    print("[PASS] Data preprocessed and split.")
    
    # 3. Training
    trainer = ModelTrainer()
    model = trainer.train_model(X_train, y_train, model_type='rf')
    metrics, _ = trainer.evaluate_model(X_test, y_test)
    print(f"[PASS] Model trained. Metrics: {metrics}")
    
    # 4. Saving
    model_path = 'SPP/models/test_model.pkl'
    trainer.save_model(model_path)
    if os.path.exists(model_path):
        print(f"[PASS] Model saved to {model_path}.")
    else:
        print("[FAIL] Model file not created.")
        sys.exit(1)
        
    # 5. Prediction
    predictor = Predictor(model_path)
    input_data = np.array([5, 80, 70, 90, 7, 2]) # Random valid inputs
    prediction = predictor.predict(input_data)
    print(f"[PASS] Prediction successful: {prediction}")
    
    # 6. Visualization
    visualizer = Visualizer(output_dir='SPP/tests/output')
    visualizer.plot_feature_importance(model, X_train.columns)
    if os.path.exists('SPP/tests/output/feature_importance.png'):
        print("[PASS] Feature importance plot created.")
    else:
        print("[FAIL] Feature importance plot not created.")
        
    print("\nVerification completed successfully!")

if __name__ == "__main__":
    verify_pipeline()
