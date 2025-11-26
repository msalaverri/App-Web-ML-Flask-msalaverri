from utils import db_connect
engine = db_connect()

import os
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar modelo, encoders y scaler
model = joblib.load('obesity_model.pkl')
le_dict = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Debug: print encoder classes to console
for col, le in le_dict.items():
    print(f"{col}: {le.classes_}")

# Opciones para el formulario (valores originales del dataset)
options = {
    "Gender": ["Female", "Male"],
    "family_history_with_overweight": ["no", "yes"],
    "FAVC": ["no", "yes"],
    "CAEC": ["Always", "Frequently", "Sometimes", "no"],
    "SMOKE": ["no", "yes"],
    "SCC": ["no", "yes"],
    "CALC": ["Always", "Frequently", "Sometimes", "no"],
    "MTRANS": ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Recoger datos del form
        features = {}
        # Categóricas
        for col in options.keys():
            value = request.form[col]
            encoded = le_dict[col].transform([value])[0]
            features[col] = encoded
        
        # Numéricas (del form)
        num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for col in num_features:
            features[col] = float(request.form[col])
        
        # DataFrame with enforced order
        feature_order = [
            'Gender',
            'Age',
            'Height',
            'Weight',
            'family_history_with_overweight',
            'FAVC',
            'FCVC',
            'NCP',
            'CAEC',
            'SMOKE',
            'CH2O',
            'SCC',
            'FAF',
            'TUE',
            'CALC',
            'MTRANS'
        ]

        df_input = pd.DataFrame([[features[col] for col in feature_order]], columns=feature_order)
        df_input[num_features] = scaler.transform(df_input[num_features])

        print("Input row for prediction:")
        print(df_input)

        # Predict
        pred = model.predict(df_input)[0]
        print("Raw prediction:", pred)
        prediction = le_dict['NObeyesdad'].inverse_transform([pred])[0]
        print("Decoded label:", prediction)
    
    return render_template('index.html', options=options, prediction=prediction)