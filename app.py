#Imports
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, jsonify, request
import keras
#from keras.models import load_model

from tensorflow.keras.models import load_model
import joblib

#Flask Setup
app = Flask(__name__)
model = load_model("models/Normal_Neural_Network_VRoX3.h5")
scaler = joblib.load("models/scaler_VRo4.save") 

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

        actual_date = datetime.today()
        user_birthday = datetime.strptime(form_data['birthday'], '%Y-%m-%d')
        delta_birthday = actual_date - user_birthday
        days_birth = delta_birthday.days
        user_employment = datetime.strptime(form_data['employment'], '%Y-%m-%d')
        delta_employment = actual_date - user_employment
        days_employed = delta_employment.days
        
        quiz_results = {'CODE_GENDER': [form_data['code_gender']], 'FLAG_OWN_CAR': [form_data['flag_own_car']], 
        'FLAG_OWN_REALTY': [form_data['flag_own_realty']], 'CNT_CHILDREN': [form_data['cnt_children']], 
        'AMT_INCOME_TOTAL': [form_data['amt_income_total']], 'NAME_INCOME_TYPE': [form_data['name_income_type']], 
        'NAME_EDUCATION_TYPE': [form_data['name_education_type']], 'NAME_FAMILY_STATUS': [form_data['name_family_status']],
        'NAME_HOUSING_TYPE': [form_data['name_housing_type']], 'DAYS_BIRTH':  [days_birth], 
        'DAYS_EMPLOYED': [days_employed], 'FLAG_MOBIL': [form_data['flag_mobil']], 
        'FLAG_WORK_PHONE': [form_data['flag_work_phone']], 'FLAG_PHONE': [form_data['flag_phone']], 
        'FLAG_EMAIL': [form_data['flag_email']], 'CNT_FAM_MEMBERS': [form_data['cnt_fam_members']]}

        # quiz_df = pd.DataFrame.from_dict(quiz_results)

        # for column in quiz_df:
        #     quiz_df[column] = quiz_df[column].astype(int)

        # quiz_transformed = scaler.transform(quiz_df)

        # prediction = np.rint(model.predict(quiz_transformed))

        # return render_template('prediction.html', form_data = form_data, prediction_text = 'Predicted Class: {}'.format(prediction))

        quiz_df = pd.DataFrame.from_dict(quiz_results)
        for column in quiz_df:
            quiz_df[column] = quiz_df[column].astype(int)
        quiz_transformed = scaler.transform(quiz_df)
        prediction = np.rint(model.predict(quiz_transformed))
        predicted_class = '{}'.format(prediction)

        if predicted_class == "[[1. 0. 0.]]":
             final_prediction = str("The user does not represent a risk: The input information of the user indicates that they have sufficient assets and will be a low risk. A credit card or credit could be given to this user.")
        elif predicted_class == "[[0. 1. 0.]]":
             final_prediction = str("The user represents a risk: The input information of the user indicates that the client could have sufficient assets, but other parameters increase the risk for them. This specific case needs to be further analyzed by your credit provider.")
        elif predicted_class == "[[0. 0. 1.]]":
             final_prediction = str("This user represents a high risk: No credit card or credit should be given to this user.")
        else:
             final_prediction = str("We don't have enough data to define your risk as a user.")
        return render_template('prediction.html', form_data = form_data, days_birth = days_birth, days_employed = days_employed,
        prediction_text = final_prediction)

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