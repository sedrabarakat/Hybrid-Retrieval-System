import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from storage.vector_storage import load_tfidf_matrix, load_doc_ids
from indexing.inverted_index_loader import load_inverted_index
from query_processing import QueryProcessor

def match_and_rank(query: str, dataset_name: str, similarity_threshold=0.0001, top_k=None):
    qp = QueryProcessor(dataset_name)
    query_vector, tokens = qp.process(query)

    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    doc_ids = load_doc_ids(f"{dataset_name}_all")
    inverted_index = load_inverted_index(dataset_name)

    candidate_doc_indices = set()
    for token in tokens:
        if token in inverted_index:
            candidate_doc_indices.update(inverted_index[token])

    if not candidate_doc_indices:
        return OrderedDict()

    candidate_doc_indices = sorted(candidate_doc_indices)
    candidate_doc_vectors = tfidf_matrix[candidate_doc_indices]

    similarity_scores = cosine_similarity(query_vector, candidate_doc_vectors).flatten()

    ranking = {
        doc_ids[i]: float(score)
        for i, score in zip(candidate_doc_indices, similarity_scores)
        if score >= similarity_threshold
    }

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_ranking = sorted_ranking[:top_k]

    return OrderedDict(sorted_ranking)
