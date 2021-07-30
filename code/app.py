import numpy as np
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
import pandas as pd
import random
import logging
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-base")
labels_mapping = ['contradiction','entailment','neutral']


@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Text to be tested" />
        <input type="submit" />'''



def load_data():
    corpus_path = "corpus.jsonl"
    query_path = "queries.jsonl"
    qrels_path = "qrels.tsv"

    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path, 
        query_file=query_path, 
        qrels_file=qrels_path).load_custom()

    return corpus, queries, qrels



def get_all_results():
    corpus, queries, qrels = load_data()
    #### Provide parameters for elastic-search
    hostname = "localhost" 
    index_name = "ourdata" 
    initialize = False # True, will delete existing index with same name and reindex all documents

    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)
    return results, corpus, queries, qrels




def get_top10_results():
    results, corpus, queries, qrels = get_all_results()
    top_k=10
    query_id, ranking_scores = random.choice(list(results.items()))
    print(query_id)
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    print("Query : %s\n" % queries[query_id])
    doc_ids = []
    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        doc_ids.append(doc_id)
        # print(corpus[doc_id].get('text'))
        passage_text = corpus[doc_id].get('text')
        enc = tokenizer(queries[query_id],passage_text,return_tensors='pt',truncation=True)
        with torch.no_grad():
            label = torch.argmax(model(**enc).logits).item()
        
        print(doc_id+'-------------',label)
    
    print('\n')
    print(corpus[doc_ids[0]].get('text'))


@app.post('/predict')
def predict(text:str = Form(...)):
    clean_text = my_pipeline(text) #clean, and preprocess the text through pipeline
    loaded_model = tf.keras.models.load_model('sentiment.h5') #load the saved model 
    predictions = loaded_model.predict(clean_text) #predict the text
    sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
    probability = max(predictions.tolist()[0]) #calulate the probability
    if sentiment==0:
         t_sentiment = 'negative' #set appropriate sentiment
    elif sentiment==1:
         t_sentiment = 'neutral'
    elif sentiment==2:
         t_sentiment='postive'
    return { #return the dictionary for endpoint
         "ACTUALL SENTENCE": text,
         "PREDICTED SENTIMENT": t_sentiment,
         "Probability": probability
    }




