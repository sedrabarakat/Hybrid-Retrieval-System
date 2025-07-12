from collections import OrderedDict
import numpy as np
from collections import OrderedDict
from .embedding_query_processing import EmbeddingQueryProcessor

import faiss

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def match_and_rank_embedding(query_text: str, dataset_name: str, similarity_threshold=0.3, top_k=10):
    processor = EmbeddingQueryProcessor(dataset_name, model_name='all-mpnet-base-v2')
    query_embedding, tokens = processor.process(query_text)

    embeddings = processor.embeddings
    doc_ids = processor.doc_ids

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores = np.dot(embeddings, query_embedding.T).squeeze()

    filtered = [
        (doc_ids[i], float(scores[i]))
        for i in range(len(scores))
        if scores[i] >= similarity_threshold
    ]

    ranked = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

    return  OrderedDict(ranked)


def match_and_rank_faiss(query_text: str, dataset_name: str, top_k: int = 10):
    processor = EmbeddingQueryProcessor(dataset_name, model_name='all-mpnet-base-v2')
    query_embedding, tokens = processor.process(query_text)
    
    if query_embedding.ndim == 1:query_embedding = query_embedding.reshape(1, -1)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vector_store", "vector_store_index"))
    path = os.path.join(base_dir, f"{dataset_name}_faiss_index.index")
    index = faiss.read_index(path)

    
    scores, indices = index.search(query_embedding, top_k)  # top_k أقرب عناصر

    
    doc_ids = processor.doc_ids

    
    ranked_results = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx == -1: 
            continue
        score = float(scores[0][i])
        doc_id = doc_ids[idx]
        ranked_results.append((doc_id, score))

    return OrderedDict(ranked_results)
