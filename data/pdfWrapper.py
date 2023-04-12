import pandas as pd
from tqdm import tqdm
import os
from PyPDF2 import PdfReader
from typing import Union
import io
try:
    import camelot

except ImportError:
    raise ValueError ('please install camelot')

class PdfWrapper:

    

    def __init__(self, stream:Union[str, bytes], table_thresh:float=0):
        self.stream = stream #could be bytes or str path
        self.number_of_pages = 0
        self.table_thresh = table_thresh

        if isinstance(self.stream, io.BytesIO):
            self.path_temp = self._create_tempfile()
        else:
            self.path_temp = stream
        print(self.path_temp)
        self.list_pages = []
        self.list_tables = []

    def _change_df(self, df: pd.DataFrame):
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        return df

    def _create_tempfile(self):
        try:
            import tempfile
        except ImportError:
            raise ValueError ('please install tempfile')

        fd, self.path_temp = tempfile.mkstemp(prefix="datatemp",suffix=".pdf")
        try:
            with os.fdopen(fd, 'bw') as tmp:
                # do stuff with temp file
                tmp.write(self.stream.getbuffer().tobytes())
        finally:
            tmp.close()
        return self.path_temp

    def extract_texts(self)->list[str]:
        pdf = PdfReader(stream = self.stream)

        self.number_of_pages = len(pdf.pages)
        print("Read pages started ...")
        for nb in tqdm(range(self.number_of_pages), desc="Pages lookup"):
            parts = []
            page = pdf.pages[nb]    ## Extracting information
            parts.append(page.extract_text())
            text_body = "".join(parts)
            self.list_pages.append(text_body)
        print("Read pages ended ...")
        print(f'number of pages is {self.number_of_pages}')
        pdf_text = '\n'.join(self.list_pages)
        return pdf_text

    def extract_tables(self)->list[str]:
        if self.number_of_pages == 0:
            pdf = PdfReader(self.path_temp)
            self.number_of_pages = len(pdf.pages)
        
        for nb in tqdm(range(0, self.number_of_pages), desc="Table lookup"):
            # try:
                tables = camelot.read_pdf(self.path_temp, pages=str(nb))
                for i in range(len(tables)):
                    if tables[i].parsing_report['accuracy']>self.table_thresh:
                        df = tables[i].df
                        df = self._change_df(df)
                        self.list_tables.append(df.to_markdown())  
            # except:
            #     print('no')
            #     continue
        text_tables = '\n'.join(self.list_tables)
        return text_tables
      
        
    
