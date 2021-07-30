import numpy as np
from fastapi import FastAPI, Form
import pandas as pd




from beir.datasets.data_loader import GenericDataLoader

corpus_path = "corpus.jsonl"
query_path = "queries.jsonl"
qrels_path = "qrels.tsv"

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()

