import pandas as pd

def change_df(df):
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df
def define_replacement(df):
    
    print(df.to_markdown())

def replace_text_with_table(text:str, df):
    try:
        last_row = " ".join(list(df.iloc[len(df)-1]))
        index_column1 = text.find(df.columns[0])
        index_last_row = text.find(last_row)
        text =text.replace(text[index_column1: index_last_row+len(last_row)],f"\n{df.to_markdown()}\n")
 
        return text
    except:
        print("Table cannot be found in text, will not be included")
        pass