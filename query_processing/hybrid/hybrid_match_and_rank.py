import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from hybrid.hybrid_query_process import process_hybrid_query
from storage.vector_storage import load_hyprid_ids
from vectorize.hybrid import generate_hybrid


def match_and_rank_hybrid(query_text: str, dataset_name: str, top_k=10, similarity_threshold=0.3):

    doc_ids=load_hyprid_ids(dataset_name=dataset_name)

    hybrid_vectors=generate_hybrid(dataset_name=dataset_name)
    query_vector, tokens = process_hybrid_query(query_text, dataset_name)

    scores = cosine_similarity(query_vector, hybrid_vectors).flatten()

    ranked_indices = np.argsort(scores)[::-1]

    ranked = [(doc_ids[i], float(scores[i])) for i in ranked_indices[:top_k]]

    filtered = [(str(doc_id), score) for doc_id, score in ranked if score >= similarity_threshold]

    return OrderedDict(filtered)
