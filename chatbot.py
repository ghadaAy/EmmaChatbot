from embeddings.similarity import order_document_sections_by_query_similarity
from utils.utils import read_context_embeddings
from transformers import GPT2TokenizerFast
import pandas as pd
from embeddings.compute_embeddings import get_doc_embedding
from utils.utils import read_cfg_file
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

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, memory: str) -> str:
    """
    Fetch relevant 
    """
    # memory_tokenization = tokenizer.tokenize(memory)
    # j=0
    # if len(memory_tokenization)>MAX_COMPLETION_TOKENS-MAX_SECTION_LEN:
    #     while len(tokenizer.tokenize(memory))>MAX_COMPLETION_TOKENS-MAX_SECTION_LEN:
    #         j+=1
    #         memory_list = memory.split("\n")
    #         memory = "\n".join(memory[j:])
            
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[int(section_index)]
        chosen_sections_len += document_section.tokens + separator_len

        if chosen_sections_len > MAX_LEN_PROMPT-len(tokenizer.tokenize(HEADER))-len(tokenizer.tokenize( "\n\n Q: " + question + "\n A:")):
            break

        # Iterate over series using Series.iteritems()
        # for _, values in enumerate(document_section.text):
        chosen_sections.append(SEPARATOR + document_section.text.replace("\n", " "))
            #print('index: ', indx, 'value: ', values)

        #chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(section_index)
            
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
    
    
    return HEADER +"".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict,
    memory: str = "",
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df,
        memory
    )
    
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    answer = response["choices"][0]["text"].strip(" \n")
    return answer

    #return response["choices"][0]["text"].strip(" \n")

if __name__=="__main__":
    context_embeddings = read_context_embeddings("embeddings/context_embeddings.json")
    df = pd.read_csv('.\data\df_tokenized.csv')
 
    resulat = []
    #memory = []
    memory=""
    print("HELLO I'AM METACHATBOT, ASK ME ABOUT THE DATASET, I'LL DO MY BEST TO HELP")
    i=0
    while True:
    #for _,row in df_test.iterrows():
        #question = row["question"] 
        i+=1
        question=input(">> ")
        
        #answer = answer_query_with_context(question, df, context_embeddings)
        answer = answer_query_with_context(question, df, context_embeddings, memory)
        memory = memory+f"question{i}={question}\n answer{i}={answer}\n"
        print("===\n", answer)
        #memory.append((question, answer))

 
