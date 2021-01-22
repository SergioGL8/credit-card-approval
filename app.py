#Imports
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, jsonify, request
import keras
from keras.models import load_model

from tensorflow.keras.models import load_model
from sklearn.externals import joblib

#Flask Setup
app = Flask(__name__)
model = load_model("models/Normal_Neural_Network_V3.h5")
scaler = joblib.load("models/scaler.save") 

#Flask Routes

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route("/prediction", methods=['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        return f"The URL /prediction was accessed directly. Try going to '/quiz' to submit form"
    if request.method == 'POST':
        
        form_data = request.form
        
        quiz_results = {'CODE_GENDER': [form_data['code_gender']], 'FLAG_OWN_CAR': [form_data['flag_own_car']], 
        'FLAG_OWN_REALTY': [form_data['flag_own_realty']], 'CNT_CHILDREN': [form_data['cnt_children']], 
        'AMT_INCOME_TOTAL': [form_data['amt_income_total']], 'NAME_INCOME_TYPE': [form_data['name_income_type']], 
        'NAME_EDUCATION_TYPE': [form_data['name_education_type']], 'NAME_FAMILY_STATUS': [form_data['name_family_status']],
        'NAME_HOUSING_TYPE': [form_data['name_housing_type']], 'DAYS_BIRTH': [form_data['days_birth']], 
        'DAYS_EMPLOYED': [form_data['days_employed']], 'FLAG_MOBIL': [form_data['flag_mobil']], 
        'FLAG_WORK_PHONE': [form_data['flag_work_phone']], 'FLAG_PHONE': [form_data['flag_phone']], 
        'FLAG_EMAIL': [form_data['flag_email']], 'CNT_FAM_MEMBERS': [form_data['cnt_fam_members']]}

        quiz_df = pd.DataFrame.from_dict(quiz_results)

        for column in quiz_df:
            quiz_df[column] = quiz_df[column].astype(int)

        quiz_transformed = scaler.transform(quiz_df)

        prediction = np.rint(model.predict(quiz_transformed))

        return render_template('prediction.html', form_data = form_data, prediction_text = 'Predicted Class: {}'.format(prediction))

"""
        keys = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 
        'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS']

        values =[form_data['code_gender'], form_data['flag_own_car'], form_data['flag_own_realty'], form_data['cnt_children'], 
        form_data['amt_income_total'], form_data['name_income_type'], form_data['name_education_type'], form_data['name_family_status'], form_data['name_housing_type'],
        form_data['days_birth'], form_data['days_employed'], form_data['flag_mobil'], form_data['flag_work_phone'], form_data['flag_phone'],
        form_data['flag_email'], form_data['occupation_type'], form_data['cnt_fam_members']]
"""

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)