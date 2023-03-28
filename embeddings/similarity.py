from .compute_embeddings import get_doc_embedding
import numpy as np

def vector_similarity(x: list, y: list)->float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    if len(y)>len(x):
        x = np.pad(x, (0,len(y)-len(x)),  'constant', constant_values=(0,0))
    elif len(y)<len(x):
        y = np.pad(y, (0,len(x)-len(y)), 'constant', constant_values=(0,0))

    return np.dot(x, y)

def order_document_sections_by_query_similarity(query: str, contexts: dict)->list:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_doc_embedding(query)
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities