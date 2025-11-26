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


# Opciones para el formulario (valores originales del dataset)
options = {
    'Gender': ['Male', 'Female'],
    'family_history_with_overweight': ['yes', 'no'],
    'FAVC': ['yes', 'no'],  # Food between meals
    'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],  # Consume alcohol
    'SMOKE': ['yes', 'no'],
    'SCC': ['yes', 'no'],  # Calories monitoring
    'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],  # Consume alcohol
    'MTRANS': ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking']
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Recoger datos del form
        features = {}
        # Categóricas
        for col, vals in options.items():
            value = request.form[col]
            encoded = le_dict[col].transform([value])[0]
            features[col] = encoded
        
        # Numéricas (del form)
        num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for col in num_features:
            features[col] = float(request.form[col])
        
        # DataFrame para predecir
        df_input = pd.DataFrame([features])
        df_input[num_features] = scaler.transform(df_input[num_features])
        
        # Predicción
        pred = model.predict(df_input)[0]
        pred_label = le_dict['NObeyesdad'].inverse_transform([pred])[0]
        prediction = pred_label
       
    
    return render_template('index.html', options=options, prediction=prediction)