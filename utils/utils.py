import pandas as pd 
import pickle
import json


def read_context_embeddings(path:str = "embeddings/context_embeddings.pkl")->dict:
    if path.endswith('pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f) 

    if path.endswith('json'):
        with open(path, 'rb') as f:
            data = json.load(f)
    return data