#Imports
from flask import Flask, render_template, redirect, jsonify, request
import numpy as np
##from tensorflow.keras.models import load_model

#Flask Setup
app = Flask(__name__)
##model = load_model("xxxxx")

#Flask Routes

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/credit")
def credit():
    return render_template("credit.html")

@app.route("/prediction", methods=["POST"])
def predict(request):

    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    ##prediction = model.predict(final_features)

    return render_template("credit.html", prediction_text = "Prediction: {}".format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
