import joblib
import os

def load_model(path_to_models):
    # 'model_selection.txt' contains the name of the joblib file we want to use
    with open(os.path.join(path_to_models, 'model_selection.txt'), "r") as model_selection_file :
        model_dump = model_selection_file.read()

    # Loads a model previously saved 
    model = joblib.load(os.path.join(path_to_models, model_dump))
    
    return model