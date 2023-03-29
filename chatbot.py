from embeddings.similarity import order_document_sections_by_query_similarity
from utils.utils import read_context_embeddings
from transformers import GPT2TokenizerFast
import pandas as pd
from embeddings.compute_embeddings import get_doc_embedding
from configparser import ConfigParser
import openai
from configparser import ConfigParser


configur = ConfigParser()
configur.read('cfg/cfg.ini')

openai.api_key = configur.get('API','OPEN_AI_API')
COMPLETIONS_MODEL = configur.get('MODELS','COMPLETIONS_MODEL')
MODEL_NAME = configur.get('MODELS','MODEL_NAME')
MAX_COMPLETION_TOKENS = int(configur.get('MODELS','MAX_COMPLETION_TOKENS'))

MAX_LEN_PROMPT = 4000-MAX_COMPLETION_TOKENS
DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
SEPARATOR = "\n* "
HEADER = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know. Can you try asking the question differently?"\n\nContext:\n"""
    
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

f"Context separator contains {separator_len} tokens"
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": MAX_COMPLETION_TOKENS,
    "model": COMPLETIONS_MODEL,
}

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    df = df.fillna('')
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    for _, section_index in most_relevant_document_sections:
        if len(df.loc[int(section_index)].text)<3:
            pass
        else:
            document_section = df.loc[int(section_index)]
            chosen_sections_len += document_section.tokens + separator_len
            if chosen_sections_len > MAX_LEN_PROMPT-len(tokenizer.tokenize(HEADER))-len(tokenizer.tokenize( "\n\n Q: " + question + "\n A:")):
                break
            chosen_sections.append(SEPARATOR + document_section.text.replace("\n", " "))
            chosen_sections_indexes.append(section_index)
    
    return HEADER +"".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict,
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)
    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    answer = response["choices"][0]["text"].strip(" \n")
    return answer

def answer_question(question: str)->str:
    context_embeddings = read_context_embeddings("embeddings/context_embeddings.json")
    df = pd.read_csv('.\data\df_tokenized.csv')    
    answer = answer_query_with_context(question, df, context_embeddings)
    return answer
