## Import
from PyPDF2 import PdfReader
import camelot
from table_utils import change_df, replace_text_with_table, define_replacement
import copy
import csv
from tqdm import tqdm
from transformers import GPT2TokenizerFast

pdf_file = "data/PDFs/brochure_fiscalite_francaise.pdf"
THRESH = 0
list_pages = []
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tables_text = []
def tokenize(list_txts):
    txt_token_size = []
    for txt in tqdm(list_txts):
        txt_token_size.append([txt,len(tokenizer.tokenize(txt, truncation=True, max_length=5000))])
    return txt_token_size

def visitor_body(text, cm, tm, fontDict, fontSize):
    y = tm[5]
    if y > 120 and y < 800:
        parts.append(text)

## Setup

with open(pdf_file, "rb") as f:
    pdf = PdfReader(f)
    info = pdf.metadata
    number_of_pages = len(pdf.pages)
    print("Read pages started ...")
    for nb in range(7,number_of_pages):
        parts = []
        page = pdf.pages[nb]    ## Extracting information
        page.extract_text(visitor_text=visitor_body)
        text_body = "".join(parts)
        list_pages.append(text_body)
    print("Read pages ended ...")  
    for nb in range(7,number_of_pages):
        try:
            tables = camelot.read_pdf(pdf_file, pages=str(nb))
            if len(tables)!=0:
                for i in range(len(tables)):
                    if tables[i].parsing_report['accuracy']>THRESH:

                        df = tables[i].df
                        df = change_df(df)
                        print(nb,df.to_markdown())
                        tables_text.append(df.to_markdown())
        except:
            continue
                # text_with_table = replace_text_with_table(list_pages[tables[i].parsing_report['page']-1], df)

                # list_pages[tables[i].parsing_report['page']-1] = text_with_table


tables_and_texts = list_pages+tables_text
with open("./data/PDFs/brochure_fiscalite_francaise.txt",'w',encoding="utf-8") as f:
    f.writelines(tables_and_texts)

texts = '\n'.join(tables_and_texts)
list_splits = texts.split('.')

list_splits_tokenized = tokenize(list_splits)

with open('data/brochure_fiscalite_francaise_tokenized.csv', 'w', encoding='utf-8') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(['text','tokens'])
    write.writerows(list_splits_tokenized)
    