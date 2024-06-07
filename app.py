import joblib
import os
import sys
import pandas as pd
from flask import Flask, render_template, jsonify, request
import shap

sys.path.append('./functions')
from my_functions import load_model
from my_functions import load_background

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
    json_input = request.get_json()
   
    data = pd.DataFrame(json_input['data'], index=[json_input['id']])
    for i, col in enumerate(data.columns) :
        if data.iloc[0, i] == '' :
            data[col] = None
            data[col] = pd.to_numeric(data[col])
            
    # Response part 1 : probability of default
    # [0] because it is launched on a single client
    # [1] because we want to extract the probability of the client to be a bad client
    proba_echec = model.predict_proba(data)[0][1] 

    # Response part 2 : text label
    if proba_echec > 0.5 :
        classe = 'refusé'
    else :
        classe = 'accepté'

    # Response part 3 : shap values
    shap_bg = load_background('./models')
    explainer = shap.TreeExplainer(model, shap_bg, feature_names=shap_bg.columns, model_output='probability')
    shap_values = explainer(data)

    # Returns the response object as json
    return jsonify({
        'classe': classe, 
        'proba_echec': proba_echec,
        'shap_values' : shap_values[0].values.tolist(),
        'shap_base_values' : shap_values[0].base_values
    })



if __name__ == '__main__':
    app.run()














