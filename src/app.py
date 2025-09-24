from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)

# Load the model
MODEL_PATH = Path(__file__).parent.parent / "models" / "final_model.pkl"
model = joblib.load(MODEL_PATH)

def feat_eng(df):
    """Feature engineering function copied from make_your_predictions.py"""
    df['AgeCat'] = pd.cut(df['age'],
                           bins=[-np.inf, 1, 12, 18, 65, np.inf],
                           labels=['infant', 'child', 'teenager', 'adult', 'older_adult'])

    df['BMICat'] = pd.cut(df['bmi'],
                            bins=[-np.inf, 18.5, 25, 30, np.inf],
                            labels=['underweight', 'normal', 'overweight', 'obese'])

    df['GlucoseCat'] = pd.cut(df['blood_glucose_level'],
                                bins=[-np.inf, 140, 200, np.inf],
                                labels=['normal', 'impaired', 'diabetic'])
    
    df['HbA1cCat'] = pd.cut(df['HbA1c_level'],
                            bins=[0, 5.6, 6.4, np.inf],
                            labels=['normal', 'prediabetic', 'diabetic'])
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': request.form['hypertension'],
            'heart_disease': request.form['heart_disease'],
            'smoking_history': request.form['smoking_history'],
            'bmi': float(request.form['bmi']),
            'HbA1c_level': float(request.form['HbA1c_level']),
            'blood_glucose_level': float(request.form['blood_glucose_level'])
        }
        
        # Create DataFrame and apply feature engineering
        df = pd.DataFrame([data])
        df = feat_eng(df)
        
        # Make prediction
        probability = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]
        
        # Get feature categories for display
        categories = df.loc[:, 'AgeCat':'HbA1cCat'].iloc[0].to_dict()
        
        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(probability * 100, 2),
            categories=categories
        )
        
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)