import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

class Visualizer:
    def __init__(self, output_dir='SPP/docs/diagrams'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_feature_importance(self, model, feature_names):
        """Plots feature importance for Random Forest models."""
        if not hasattr(model, 'feature_importances_'):
            print("Model does not support feature importance plotting (e.g., Linear Regression).")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath)
        print(f"Feature importance plot saved to {filepath}")
        plt.close()

    def plot_actual_vs_predicted(self, y_test, y_pred):
        """Plots actual vs predicted values."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Performance')
        
        filepath = os.path.join(self.output_dir, 'actual_vs_predicted.png')
        plt.savefig(filepath)
        print(f"Actual vs Predicted plot saved to {filepath}")
        plt.close()
