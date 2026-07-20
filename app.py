from flask import Flask, render_template, request, send_from_directory
import pickle
import numpy as np

app = Flask(__name__)

logistic = pickle.load(open("Model/logistic.pkl", "rb"))
decision_tree = pickle.load(open("Model/decision_tree.pkl", "rb"))
knn = pickle.load(open("Model/knn.pkl", "rb"))
svm = pickle.load(open("Model/svm.pkl", "rb"))

scaler = pickle.load(open("Model/scaler.pkl", "rb"))


@app.route("/style.css")
def style():
    return send_from_directory("templates", "style.css", mimetype="text/css")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    study = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    grades = float(request.form["grades"])
    extra = int(request.form["extra"])
    education = request.form["education"]

    bachelor = 1 if education == "Bachelor" else 0
    doctorate = 1 if education == "Doctorate" else 0
    high_school = 1 if education == "High School" else 0
    master = 1 if education == "Master" else 0

    features = [[
        study,
        attendance,
        grades,
        extra,
        bachelor,
        doctorate,
        high_school,
        master
    ]]

    features = scaler.transform(features)

    result = {
        "Logistic Regression": logistic.predict(features)[0],
        "Decision Tree": decision_tree.predict(features)[0],
        "KNN": knn.predict(features)[0],
        "Support Vector Machine": svm.predict(features)[0]
    }

    return render_template("result.html", result=result)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)