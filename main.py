import os
import sys
import numpy as np
import pandas as pd

# Add current directory to path so we can import spp package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spp.data_loader import DataLoader
from spp.preprocessor import Preprocessor
from spp.trainer import ModelTrainer
from spp.predictor import Predictor
from spp.visualizer import Visualizer

DATA_PATH = 'data/student_data.csv'
MODEL_PATH = 'models/student_performance_model.pkl'

def main():
    print("===========================================")
    print("   Student Performance Predictor (SPP)   ")
    print("===========================================")
    
    while True:
        print("\nOptions:")
        print("1. Train Model")
        print("2. Predict Performance")
        print("3. View Visualizations")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            print("\n--- Training Model ---")
            try:
                # Load Data
                loader = DataLoader(DATA_PATH)
                df = loader.load_data()
                loader.validate_data(df)
                print("Data loaded successfully.")
                
                # Preprocess
                preprocessor = Preprocessor()
                df = preprocessor.clean_data(df)
                X_train, X_test, y_train, y_test = preprocessor.split_data(df)
                
                # Train
                trainer = ModelTrainer()
                print("Training Random Forest model...")
                model = trainer.train_model(X_train, y_train, model_type='rf')
                
                # Evaluate
                metrics, y_pred = trainer.evaluate_model(X_test, y_test)
                print(f"Model Evaluation Metrics: {metrics}")
                
                # Save
                trainer.save_model(MODEL_PATH)
                print(f"Model saved to {MODEL_PATH}")
                
                # Visualize
                visualizer = Visualizer()
                visualizer.plot_actual_vs_predicted(y_test, y_pred)
                visualizer.plot_feature_importance(model, X_train.columns)
                
            except Exception as e:
                print(f"Error during training: {e}")
                
        elif choice == '2':
            print("\n--- Predict Performance ---")
            try:
                predictor = Predictor(MODEL_PATH)
                
                print("Enter student details:")
                study_hours = float(input("Study Hours (1-10): "))
                attendance = float(input("Attendance % (50-100): "))
                prev_scores = float(input("Previous Scores (40-100): "))
                assignment = float(input("Assignment Completion % (0-100): "))
                sleep = float(input("Sleep Hours (4-10): "))
                social = float(input("Social Media Usage (0-5): "))
                
                input_data = np.array([study_hours, attendance, prev_scores, assignment, sleep, social])
                
                prediction = predictor.predict(input_data)
                print(f"\nPredicted Performance Index: {prediction:.2f}")
                
                if prediction >= 80:
                    print("Category: Excellent")
                elif prediction >= 60:
                    print("Category: Good")
                elif prediction >= 40:
                    print("Category: Average")
                else:
                    print("Category: Needs Improvement")
                    
            except FileNotFoundError:
                print("Model not found. Please train the model first (Option 1).")
            except Exception as e:
                print(f"Error during prediction: {e}")

        elif choice == '3':
            print("\n--- Visualizations ---")
            if os.path.exists('SPP/docs/diagrams/actual_vs_predicted.png'):
                print("Visualizations are saved in SPP/docs/diagrams/")
                print("- actual_vs_predicted.png")
                print("- feature_importance.png")
                # In a real GUI we would show them, here we just notify
            else:
                print("No visualizations found. Train the model first.")
                
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
