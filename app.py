#Imports
import numpy as np
from flask import Flask, render_template, redirect, jsonify, request



#Flask Setup
app = Flask(__name__)

#Flask Routes

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/credit')
def credit():
    return render_template('credit.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    #prediction = model.predict(final_features) # making prediction

    return render_template('credit.html', prediction_text='Predicted Class: {}'.format(final_features)) # rendering the predicted result

if __name__ == '__main__':
    app.run(debug=True)