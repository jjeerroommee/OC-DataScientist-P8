import joblib
import os
import sys
import pandas as pd
from flask import Flask, render_template, jsonify, request

sys.path.append('./functions')
from my_functions import load_model

app = Flask(__name__)

@app.route('/')
def home():
    # Le guide d'utilisation de l'API est accessible sur la page d'accueil en-dessous :
    return render_template('index.html')


@app.route('/predict/', methods=['POST'])
def classif():
    #print(request.full_path)
        
    # Loads a model previously saved 
    model = load_model('./models')
        
    # Gets a dataframe of features from the post request's body
    data_json = request.get_json()
    data = pd.DataFrame(data_json['inputs'])
    
    # Creates a response object
    # [0] because it is launched on a single client
    # [1] because we want to extract the probability of the client to be a bad client
    proba_echec = model.predict_proba(data)[0][1] 

    if proba_echec > 0.5 :
        classe = 'refusé'
    else :
        classe = 'accepté'
    
    # Returns the response object as json
    return jsonify({
        'classe': classe, 
        'proba_echec': proba_echec
    })


if __name__ == '__main__':
    app.run()














