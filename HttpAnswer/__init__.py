import logging
from urllib.error import HTTPError
import json
from chatbot import answer_question
import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    
    try:
        req_body = req.get_json()
        question = req_body.get('question')
        if not question:
            return func.HttpResponse(
                headers={'Content-Type': 'application/json'},
                status_code=500,
                body=  json.dumps({
                "message": "question is needed"
                })
            )
        else:
        
            #answer = answer_query_with_context(question, df, context_embeddings)
            answer = answer_question(question)
            return func.HttpResponse(
                headers={'Content-Type': 'application/json'},
                status_code=200,
                body=  json.dumps({
                "answer": answer
                })
            )
    except HTTPError as error :
        return func.HttpResponse(
                error.info,
                status_code=500
            )
  
  

    
