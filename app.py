
from flask import Flask,request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)
CORS(app)

@app.route('/predict',methods=['POST'])
def index():

    data_r = request.get_json()
    

    formatted = pd.DataFrame(data=data_r, index=[0])

    return jsonify({'Value': model.predict(formatted)[0] })


if __name__ == "__main__": 

    app.run(debug=True,port=5000)

