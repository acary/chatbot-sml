# Serve model as a flask application

import pickle
import numpy as np
import datetime
from flask import Flask, request, Flask, render_template, request, jsonify, Blueprint, \
    request, flash, g, session, redirect, url_for, send_file, \
    send_from_directory, make_response
import ast
import json

model = None
app = Flask(__name__)

def load_model():
    ''' Load prediction model '''
    global model
    # model variable refers to the global variable
    with open('iris_trained_model.pkl', 'rb') as f:
        model = pickle.load(f)


@app.errorhandler(404)
def not_found(error):
    '''Render 404 error page if resource not found.'''
    return 'Not found (404)'


@app.route('/')
def home():
    ''' Main entrypoint '''
    return render_template('chat.html')


@app.route("/create-entry", methods=["POST"])
def create_entry():
    ''' Process and analyze new entry '''

    load_model()

    req = request.get_json()

    print(req)

    data = req['message']
    data = list(data.split(" "))
    data = [float(i) for i in data]
    data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
    prediction = model.predict(data)  # runs globally loaded model on the data
    y = str(prediction[0])
    print(y)

    res = make_response(jsonify(y), 200)

    return res


if __name__ == '__main__':

    app.run(host='127.0.0.1', port=8080)
