from flask import Flask
from flask import request
from joblib import load

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
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}