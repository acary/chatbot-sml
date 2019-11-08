# Serve model as a flask application

import pickle
import numpy as np
import datetime
from flask import Flask, request, Flask, render_template, request, jsonify, Blueprint, \
    request, flash, g, session, redirect, url_for, send_file, \
    send_from_directory
import ast
import json

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('iris_trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

@app.errorhandler(404)
def not_found(error):
    '''Render 404 error page if resource not found.'''
    return 'Not found (404)'

@app.route('/')
def home_endpoint():

    return render_template('chat.html')


@app.route('/predict', methods=['POST'])
def get_prediction():

    if request.method == 'POST':

        # Get input text
        x = request.form.get('input').strip()
        x = list(x.split(" "))
        x = [float(i) for i in x]
        print(x)

        # Format input text for processing
        # x = json.dumps(x)
        # x = jsonify(x)
        # x = x.data.decode("utf-8")
        # x = ast.literal_eval(x)
        print(x)
        print(type(x)) # String

        data = x # [5.9,3.0,5.1,1.8]
        print(type(data)) # List
        print(x == data)
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)

        # Make prediction
        prediction = model.predict(data)  # runs globally loaded model on the data
        y = str(prediction[0])
        print(y)

        # Return response to client
        return redirect(url_for('session', data=y))

@app.route('/session/<data>', methods=['GET'])
def session(data):

    print(data)

    return render_template('chat.html', data=data)

if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='127.0.0.1', port=8080)
