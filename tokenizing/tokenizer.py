from transformers import GPT2TokenizerFast
import pandas as pd
from tqdm import tqdm

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def split_text(tables_and_texts:list[str])->list[str]:
    texts = '\n'.join(tables_and_texts)
    list_splits = texts.split('.')
    return list_splits

def tokenize(list_txts: list[str])->list[str,int]:
    txt_token_size = []
    for txt in tqdm(list_txts, desc="Tokenizing..."):
        txt_token_size.append([txt, len(tokenizer.tokenize(txt, truncation=True, max_length=5000))])
    return txt_token_size


def create_tokenize_csv(tables_and_texts: list[str]):
    list_txts = split_text(tables_and_texts)
    tokenized_text = tokenize(list_txts)
    df = pd.DataFrame(tokenized_text,columns=['text','tokens'])
    df.to_csv('data/tokenized_csvs/pdf.csv', encoding='utf-8')
    df = df.dropna()
    print("pdf.csv is created")
    return df