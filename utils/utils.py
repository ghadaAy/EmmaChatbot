import pandas as pd 
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import json

def split_train_test(file_path:str)->list:
    df = pd.read_csv(file_path)
    train ,test = train_test_split(df, test_size=0.02)       
    return train,test


def read_context_embeddings(path:str = "embeddings/context_embeddings.pkl")->dict:
    if path.endswith('pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f) 

    if path.endswith('json'):
        with open(path, 'rb') as f:
            data = json.load(f)
    return data