# Student Performance Predictor (SPP)

A machine learning project to predict student performance based on various factors like study hours, attendance, and previous scores.

## Project Structure
```
SPP/
├── data/
│   └── student_data.csv       # Synthetic dataset
├── docs/
│   └── diagrams/              # Generated plots
├── spp/                       # Main package
│   ├── __init__.py
│   ├── data_loader.py         # Data loading logic
│   ├── preprocessor.py        # Data cleaning and splitting
│   ├── trainer.py             # Model training (Random Forest)
│   ├── predictor.py           # Prediction logic
│   └── visualizer.py          # Plotting functions
├── main.py                    # CLI Entry point
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to start the CLI:
```bash
python main.py
```

### Features
1. **Train Model**: Trains a Random Forest Regressor on the dataset and saves the model.
2. **Predict Performance**: Accepts user input for a new student and predicts their performance index.
3. **Visualizations**: Generates feature importance and actual vs predicted plots in `docs/diagrams/`.

## Model
The project uses a **Random Forest Regressor** for robust performance prediction. It evaluates the model using MAE, MSE, and R2 score.