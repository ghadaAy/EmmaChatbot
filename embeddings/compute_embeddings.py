import os
import pandas as pd
import csv
import openai
import string
from tqdm import tqdm
from statistics import mean
import nltk
import time
import numpy as np
from transformers import GPT2TokenizerFast
#from utils.utils import read_cfg_file
import json

#openai.api_key, COMPLETIONS_MODEL, MODEL_NAME = read_cfg_file(cfg_path='../cfg/cfg.ini')
openai.api_key = "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"


DOC_EMBEDDINGS_MODEL = "text-embedding-ada-002"

def get_embedding(text: str, model: str)->list:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str)->list:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame)->dict:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    d = {}
    for idx, r in tqdm(df.iterrows()):
      d[idx] = get_doc_embedding(str(r.text).replace("/n", " "))
      time.sleep(1)
      
    return d

# def load_embeddings(fname: str)->dict:
#     """
#     Read the document embeddings and their keys from a CSV.
    
#     fname is the path to a CSV with exactly these named columns: 
#         "title", "heading", "0", "1", ... up to the length of the embedding vectors.
#     """
    
#     df = pd.read_csv(fname, header=0)
#     max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
#     return {
#            (r.title, r.heading): [r[strs(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
#     }

if __name__=="__main__":
    df = pd.read_csv('data\df_tokenized.csv')
    print('embeddings began')
    context_embeddings = compute_doc_embeddings(df)
    print("saving to json")
    with open('context_embeddings.json', 'w') as f:
        json.dump(context_embeddings, f)

    print('saved')