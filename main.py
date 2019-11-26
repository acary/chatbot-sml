# Serve model as a flask application
import io
import os
import pickle
import numpy as np
import datetime
from flask import Flask, request, Flask, render_template, request, jsonify, Blueprint, \
    request, flash, g, session, redirect, url_for, send_file, \
    send_from_directory, make_response
import ast
import json

from controllers.ulmfit import *

model = None
app = Flask(__name__)

def load_model():
    ''' Load prediction model '''
    global model
    # `model` variable refers to the global variable
    with open('models/iris_trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

def map_classification(y):
    ''' Map classification code to descriptive value '''
    class_options = {
        '0': 'Iris setosa',
        '1': 'Iris virginica',
        '2': 'Iris versicolor',
    }
    return class_options[y]

@app.errorhandler(404)
def not_found(error):
    '''Render 404 error page if resource not found.'''
    return 'Not found (404)'


@app.route('/')
def home():
    ''' Main entrypoint '''
    return render_template('chat.html')


@app.route('/compare')
def compare():
    ''' Compare with another model '''
    return render_template('chatter.html')


@app.route("/create-entry", methods=["POST"])
def create_entry():
    ''' Process and analyze new entry '''

    # ULMFit
    load_ulm_model()
    uml_predict()

    ### Below is for Iris prediction ###
    # Load model
    load_model()

    # Get request as json
    req = request.get_json()
    data = req['message'] # Get message field contents

    # Pre-process message
    data = list(data.split(" "))
    data = [float(i) for i in data]
    data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)

    # Get prediction
    prediction = model.predict(data)
    y = str(prediction[0])
    # print(y)

    # Get value
    intent = map_classification(y)

    # Send response
    res = make_response(jsonify(intent), 200)
    return res


if __name__ == '__main__':

    app.run(host='127.0.0.1', port=8080, debug=True)
