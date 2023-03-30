import streamlit as st
from streamlit_chat import message
from chatbot import answer_query_with_context
from utils.utils import read_context_embeddings
import pandas as pd

@st.cache_resource
def read_embeddings():
    print("start")
    context_embeddings = read_context_embeddings("./embeddings/context_embeddings.json")
    df = pd.read_csv('.\data\df_tokenized.csv')    
    df = df.fillna('')
    print('end')
    return context_embeddings, df

context_embeddings, df = read_embeddings()

st.title("EmmaChatBot")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

user_input = st.text_input("You: ","", key=f"input")

    
if user_input:
    print('hey2')
    with st.spinner("Answering your question"):
        output = answer_query_with_context(user_input, df, context_embeddings)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')