import streamlit as st
import sys
sys.path.append('data')
sys.path.append('embeddings')
import json

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm
import io
from data import pdfWrapper
from pdfWrapper import PdfWrapper
import os
from tokenizing.tokenizer import create_tokenize_csv
from data_utils import write_file
from embeddings.compute_embeddings import compute_doc_embeddings

st.set_page_config(
    page_title="upload PDF",
    page_icon="👋",
)

st.title("UPLOAD PDF ")
st.sidebar.success("Select a page above.")

uploaded_file = st.file_uploader("Choose a file", type="pdf")
if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        stream = io.BytesIO(bytes_data)
        
        pdf_file = PdfWrapper(stream)
        with st.spinner("Parsing PDF"):
            text_pages = pdf_file.extract_texts()
            text_tables = pdf_file.extract_tables()
            write_file([text_pages, text_tables])

        with st.spinner("Tokenizing PDF"):    
            df = create_tokenize_csv([text_pages, text_tables])
            st.session_state['flag_done_upload'] = True

        # with st.spinner("Embedding"):    
        #     context_embeddings = compute_doc_embeddings(df)
        #     with open('embeddings/context_embeddings.json', 'w') as f:
        #         json.dump(context_embeddings, f)

        print('json saved')

        st.markdown("##### You can now ask any questions inside of chatbot")
