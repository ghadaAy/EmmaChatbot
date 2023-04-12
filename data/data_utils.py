import pandas as pd


def write_file(strings:list[str]):
    txt = '\n'.join(strings)
    with open("./data/PDFs/pdf.txt",'w',encoding="utf-8") as f:
        f.write(txt)
    print("pdf.txt is created")

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