# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, request, jsonify, Response
import pickle
import pandas as pd
import json
import uvicorn
from zipfile import ZipFile
from fastapi import FastAPI

from pandas.io.json import json_normalize
import requests




app = Flask(__name__)

def load_models():
    model = pickle.load(open('models/LRCSmote.obj','rb'))
    return model

@app.route("/")
def hello():
    """
    Ping the API.
    """
    return jsonify({"text":"Hello, the API is up and running..." })

@app.route('/predict', methods=['GET'])
def predict():
    # userid = user_id
    # user_id = 100006

    if 'id' in request.args:
        id = int(request.args['id'])

        # Charger les données à partir de train2.csv
        z = ZipFile("data/train2.zip")
        df = pd.read_csv(z.open('train2.csv'), encoding ='utf-8')

        # Filter la dataframe pour ne garder que l'utilisateur spécifié, enlever la colonne 'SK_ID_CURR' et transformer le résultat en dictionnaire JSON
        input_data = df[df['SK_ID_CURR'] == id].drop(['SK_ID_CURR', 'Unnamed: 0'], axis=1).to_numpy().reshape(1, -1)

        # parse input features from request
        request_json = input_data.tolist()
        print(request_json)
        # load model
        model = load_models()
        prediction = model.predict_proba(request_json)[:, 1][0]
        # Format prediction in percentage with 2 decimal points
        prediction = "The client has a " + str(round(prediction*100,2)) + "% risk of defaulting on their loan."
        print("prediction: ", prediction)

    # Return output
    else:
        return "Erreur: Pas d’identifiant fourni. Veuillez spécifier un id."
    
    return jsonify(json.dumps(str(prediction)))

if __name__ == '__main__':
    app.run(debug=True)

