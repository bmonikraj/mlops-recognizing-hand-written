from flask import Flask
from flask import request
from joblib import load
import numpy as np

app = Flask(__name__)
model_path = "5-svc.joblib"
model = load(model_path)

@app.route("/", methods=['GET'])
def hello_world():
    return "<b> Hello, World!</b>"

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y 
    return {'sum':z}



@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    predicted = model.predict([
        np.array(image)
    ])
    return {"y_predicted":int(predicted[0])}

@app.route("/match", methods=['POST'])
def match_digits():
    image_0 = request.json['image_0']
    image_1 = request.json['image_1']
    predicted_0 = model.predict([
        np.array(image_0)
    ])
    predicted_1 = model.predict([
        np.array(image_1)
    ])
    return {"macthed": str(predicted_0[0] == predicted_1[0])}