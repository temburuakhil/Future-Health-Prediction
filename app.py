import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, flash
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models and scaler
disease_models = {
    "Heart Disease": joblib.load('HeartDisease_model.pkl'),
    "Diabetes": joblib.load('Diabetes_model.pkl'),
    "Stroke": joblib.load('Stroke_model.pkl'),
    "Lung Disease": joblib.load('LungDisease_model.pkl'),
    "Liver Disease": joblib.load('LiverDisease_model.pkl')
}
scaler = joblib.load('scaler.pkl')

recommendations = {
    "Heart Disease": {
        "preventive_measures": "Maintain a healthy weight, exercise regularly, and avoid smoking.",
        "diet": "Eat a diet rich in vegetables, fruits, whole grains, and lean proteins.",
        "exercises": "Cardio exercises like walking, running, or cycling for 30 minutes daily.",
        "lab_tests": "Lipid panel, blood pressure test, ECG."
    },
    "Diabetes": {
        "preventive_measures": "Monitor blood sugar levels and maintain a healthy weight.",
        "diet": "Focus on low-carb, high-fiber foods. Avoid sugary foods.",
        "exercises": "Regular aerobic and strength training exercises.",
        "lab_tests": "Fasting blood sugar, HbA1c test."
    },
    "Stroke": {
        "preventive_measures": "Control high blood pressure and cholesterol, avoid smoking, and manage stress.",
        "diet": "Low-sodium, high-fiber foods. Include more fruits and vegetables.",
        "exercises": "Daily physical activity like walking, swimming, or cycling.",
        "lab_tests": "Blood pressure test, cholesterol test, brain MRI."
    },
    "Lung Disease": {
        "preventive_measures": "Avoid smoking, limit exposure to air pollution, and get vaccinated.",
        "diet": "Eat nutrient-rich foods, especially those that boost lung function like apples, berries, and broccoli.",
        "exercises": "Breathing exercises, light cardio, and strength training.",
        "lab_tests": "Pulmonary function test, chest X-ray, spirometry."
    },
    "Liver Disease": {
        "preventive_measures": "Avoid excessive alcohol consumption, maintain a healthy weight, and avoid hepatitis infections.",
        "diet": "Eat a balanced diet, avoid fatty foods, and reduce salt intake.",
        "exercises": "Light aerobic exercises, such as walking or swimming.",
        "lab_tests": "Liver function test, abdominal ultrasound."
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Retrieve and process input data
            age = int(request.form['age'])
            gender = 1 if request.form['gender'] == 'male' else 0
            bmi = float(request.form['bmi']) 
            bp = int(request.form['bp'].split('/')[0])  
            cholesterol = int(request.form['cholesterol'])
            smoking = 1 if request.form['cigarette'] != 'none' else 0
            diabetes = 1 if request.form['diabetes'] == 'yes' else 0
            alcohol = 1 if request.form['alcohol'] != 'none' else 0

            # Format input for prediction
            input_data = np.array([[age, gender, bmi, bp, cholesterol, smoking, diabetes, alcohol]])
            input_data_scaled = scaler.transform(input_data)

            # Predict risk for each disease using individual models
            risk = {}
            for disease, model in disease_models.items():
                prob = model.predict_proba(input_data_scaled)[:, 1][0]
                risk[disease] = f"{prob * 100:.2f}%"

            # Retrieve recommendations
            preventive_measures = {disease: recommendations[disease]["preventive_measures"] for disease in risk}
            diet = {disease: recommendations[disease]["diet"] for disease in risk}
            exercises = {disease: recommendations[disease]["exercises"] for disease in risk}
            lab_tests = {disease: recommendations[disease]["lab_tests"] for disease in risk}

            return render_template('form.html', risk=risk, preventive_measures=preventive_measures, diet=diet, exercises=exercises, lab_tests=lab_tests)

        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')
            return render_template('form.html')

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)