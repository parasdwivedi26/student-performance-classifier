import pandas as pd
import numpy as np
import os

def generate_data(num_samples=500):
    np.random.seed(42)
    
    # Generate synthetic features
    study_hours = np.random.uniform(1, 10, num_samples)
    attendance = np.random.uniform(50, 100, num_samples)
    prev_scores = np.random.uniform(40, 100, num_samples)
    assignment_completion = np.random.uniform(0, 100, num_samples)
    sleep_hours = np.random.uniform(4, 10, num_samples)
    social_media = np.random.uniform(0, 5, num_samples)
    
    # Generate target variable (Performance Index) with some noise
    # Formula: 2.5*Study + 0.3*Attendance + 0.4*Prev + 0.1*Assign + 0.5*Sleep - 1.5*Social + Noise
    target = (2.5 * study_hours + 
              0.3 * attendance + 
              0.4 * prev_scores + 
              0.1 * assignment_completion + 
              0.5 * sleep_hours - 
              1.5 * social_media + 
              np.random.normal(0, 2, num_samples))
    
    # Clip target to 0-100
    target = np.clip(target, 0, 100)
    
    df = pd.DataFrame({
        'Study Hours': study_hours.round(1),
        'Attendance (%)': attendance.round(1),
        'Previous Scores': prev_scores.round(1),
        'Assignment Completion': assignment_completion.round(1),
        'Sleep Hours': sleep_hours.round(1),
        'Social Media Usage': social_media.round(1),
        'Performance Index': target.round(1)
    })
    
    return df

if __name__ == "__main__":
    df = generate_data()
    output_path = os.path.join(os.path.dirname(__file__), 'student_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data generated successfully: {output_path}")
