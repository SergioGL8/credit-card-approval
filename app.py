#Imports
from flask import Flask, render_template, redirect, jsonify

#Flask Setup
app = Flask(__name__)

#Flask Routes

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/credit")
def credit():
    return render_template("credit.html")

if __name__ == '__main__':
    app.run(debug=True)
