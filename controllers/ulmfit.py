import pickle
import fastai
from fastai import *
import pandas as pd
import numpy as np
from functools import partial

def load_ulm_model():
    ''' Load ULM prediction model '''

    global model_ulmfit
    # `model_umlfit` variable refers to the global variable

    with open('models/ulmfit_classifier_model.pkl', 'rb') as f:
        model_ulmfit = pickle.load(f)
        # print(type(model_ulmfit))

def uml_predict():
    ''' Make ULM prediction '''
    # print(model_ulmfit)

    predict_response = [{
        "query":"What is the status of my loan application?",
        "intent":"loan application",
        "score":0.40
    }]

    meets_threshold = 0.70
    probability = float(predict_response[0].get("score")) # 0.80
    if probability > meets_threshold:
        print(predict_response[0].get("intent"))
    else:
        print("Out-of-scope, handle appropriately")
