import sys
import os

sys.path.append('./functions')
from my_functions import load_model

def test_metric_used():

    # charge le modèle utilisé par l'API
    model = load_model('./models')
    
    # vérifie que le modèle a été optimisé par rapport à la métrique 'bank_loss'
    metric = model.get_params()['metric']
    assert metric == 'bank_loss'