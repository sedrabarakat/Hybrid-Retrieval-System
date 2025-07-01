import sys
import os
import requests
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from storage.vector_storage import load_tfidf_matrix, load_doc_ids
from indexing.inverted_index_loader import load_inverted_index
from query_processing import QueryProcessor

import mysql.connector  # still used for other DB calls if needed

# Ensure your project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class IndexDataLoader:
    def __init__(self):
        self.cache = {}

    def load(self, dataset_name):
        if dataset_name in self.cache:
            print(f"[CACHE] Using cached index data for {dataset_name}")
            return self.cache[dataset_name]

        print(f"[DISK] Loading index data for {dataset_name} from disk...")
        tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
        doc_ids = load_doc_ids(f"{dataset_name}_all")
        inverted_index = load_inverted_index(dataset_name)

        self.cache[dataset_name] = {
            'tfidf_matrix': tfidf_matrix,
            'doc_ids': doc_ids,
            'inverted_index': inverted_index
        }

        return self.cache[dataset_name]


index_loader = IndexDataLoader()


def fetch_documents_text_via_api(dataset_name, doc_ids):
    """
    Fetch document texts from the FastAPI endpoint via HTTP GET request.
    """
    if not doc_ids:
        return {}

    base_url = "http://localhost:8000"  # Adjust if your API runs elsewhere
    endpoint = "/database/documents-text/"

    doc_ids_str = ",".join(str(i) for i in doc_ids)

    params = {
        "dataset_name": dataset_name,
        "doc_ids": doc_ids_str
    }

    try:
        response = requests.get(base_url + endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching documents text from API: {e}")
        return {}


def match_and_rank(query: str, dataset_name: str, similarity_threshold=0.0001, top_k=None):
    qp = QueryProcessor(dataset_name)
    query_vector, tokens = qp.process(query)

    # Load index data (cached or from disk)
    data = index_loader.load(dataset_name)

    tfidf_matrix = data['tfidf_matrix']
    doc_ids = data['doc_ids']
    inverted_index = data['inverted_index']

    matched_tokens = [t for t in tokens if t in inverted_index]

    if not matched_tokens:
        print("[!] No matching tokens found in inverted index. Returning empty result.")
        return OrderedDict()

    candidate_doc_ids = set()
    for token in matched_tokens:
        candidate_doc_ids.update(inverted_index[token])

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    candidate_indices = [
        doc_id_to_index[doc_id] for doc_id in candidate_doc_ids
        if doc_id in doc_id_to_index
    ]

    if not candidate_indices:
        print("[!] No candidate document indices found. Returning empty result.")
        return OrderedDict()

    candidate_indices = sorted(candidate_indices)
    candidate_vectors = tfidf_matrix[candidate_indices]

    similarity_scores = cosine_similarity(query_vector, candidate_vectors).flatten()

    ranking = {
        doc_ids[i]: float(score)
        for i, score in zip(candidate_indices, similarity_scores)
        if score >= similarity_threshold
    }

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_ranking = sorted_ranking[:top_k]

    top_doc_ids = [doc_id for doc_id, _ in sorted_ranking]

    # Fetch document texts via API call instead of direct DB call
    documents_texts = fetch_documents_text_via_api(dataset_name, top_doc_ids)

    print(f"Top {top_k if top_k else 'all'} ranked results with document content:")
    for rank, (doc_id, score) in enumerate(sorted_ranking[:5], 1):
        # Keys from JSON are strings, so convert doc_id to string
        text = documents_texts.get(str(doc_id), "[Text not found]")
        print(f"🔹 Rank: {rank}")
        print(f"   Doc ID: {doc_id}")
        print(f"   Score: {score:.6f}")
        print(f"   Text: {text[:200]}...")
        print("-" * 50)

    return OrderedDict(sorted_ranking)
