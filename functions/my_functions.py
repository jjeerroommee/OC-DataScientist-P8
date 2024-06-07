import joblib
import os
import pandas as pd

def load_model(path_to_models):
    # 'model_selection.txt' contains the name of the joblib file we want to use
    with open(os.path.join(path_to_models, 'model_selection.txt'), "r") as model_selection_file :
        model_dump = model_selection_file.read()

    # Loads a model previously saved 
    model = joblib.load(os.path.join(path_to_models, model_dump))
    
    return model

def load_background(path_to_models):
    # load the model's training Data : we will use it as basckground data in shap values calculation
    shap_bg = pd.read_csv(os.path.join(path_to_models, 'shap_background.csv'), sep=";")
    return shap_bg