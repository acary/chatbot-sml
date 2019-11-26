import pickle
import fastai
from fastai import *
import pandas as pd
import numpy as np
from functools import partial

def load_ulm_model():
    ''' Load ULM prediction model '''

    global model_umlfit
    # `model_umlfit` variable refers to the global variable

    with open('models/ulmfit_classifier_model.pkl', 'rb') as f:
        model_umlfit = pickle.load(f)
        print(type(model_umlfit))

def uml_predict():
    ''' Make ULM prediction '''
    print(model_umlfit)
    print(model_umlfit)
    # print(model_umlfit.predict("How long does the loan process take?"))
    # predict_uml = model_umlfit.predict("How long does the loan process take?")
    # print(str(predict_uml))
