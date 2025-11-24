# Problem Statement: Student Performance Predictor

## 1. Problem Statement
Students often struggle to understand whether their current learning habits are sufficient to achieve their desired academic outcomes. Teachers also lack a simple tool to estimate performance early. This project predicts a student's final exam performance based on input features such as attendance, study hours, assignment scores, etc.

## 2. Objectives
- Build a machine learning model to predict student performance.
- Provide a simple input/output interface for predictions.
- Analyze which factors most affect performance.
- Help students and educators make data-driven decisions.

## 3. Functional Requirements
- **FR1 — Data Input Module**: Load dataset and validate input values.
- **FR2 — Data Processing Module**: Clean missing values, split dataset into train/test.
- **FR3 — Model Training & Prediction Module**: Train Random Forest model, predict performance score.
- **FR4 — Reporting & Visualization**: Generate feature importance chart and accuracy metrics.
- **FR5 — User Interaction Interface**: CLI for user inputs and results.

## 4. Non-Functional Requirements
- **NFR1 — Usability**: Simple CLI interface.
- **NFR2 — Performance**: Prediction in under 1 second.
- **NFR3 — Maintainability**: Modular code structure.
- **NFR4 — Reliability**: Handles missing/invalid input gracefully.

## 5. System Architecture
User → Input Module (CLI) → Processing Pipeline → ML Model → Output Prediction
