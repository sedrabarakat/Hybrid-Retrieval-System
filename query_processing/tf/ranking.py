import sys
import os
import requests
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from indexing.inverted_index_loader import load_inverted_index
from tf.query_processing import QueryProcessor
import tempfile

class IndexDataLoader:
    def __init__(self):
        self.cache = {}

    def load_tfidf_matrix_via_api(self, dataset_name):
        """
        Load TF-IDF matrix from API endpoint as a temporary file and read it.
        """
        base_url = "http://localhost:8000"  # Adjust if your API runs elsewhere
        endpoint = f"/load_tfidf_matrix/{dataset_name}_all"
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp_file:
                response = requests.get(base_url + endpoint, stream=True)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                
                tmp_file_path = tmp_file.name
            
            # Load sparse matrix from the .npz file
            matrix = np.load(tmp_file_path)['matrix']
            
            # Clean up the temporary file
            os.unlink(tmp_file_path)
            
            return matrix
        except requests.RequestException as e:
            print(f"Error fetching TF-IDF matrix from API: {e}")
            raise
        except Exception as e:
            print(f"Error processing TF-IDF matrix: {e}")
            raise

    def load_doc_ids_via_api(self, name, vectorizer_type="tfidf"):
        """
        Load document IDs from the FastAPI endpoint /storage/load_doc_ids/{name}.
        """
        base_url = "http://localhost:8000"
        endpoint = f"/storage/load_doc_ids/{name}"
        params = {"vectorizer_type": vectorizer_type}
        try:
            response = requests.get(base_url + endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("doc_ids", [])
        except requests.RequestException as e:
            print(f"Error fetching doc_ids from API: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error fetching doc_ids: {e}")
            raise

    def load(self, dataset_name):
        """
        Load all index-related data (TF-IDF matrix, doc IDs, inverted index),
        caching results for repeated calls.
        """
        if dataset_name in self.cache:
            print(f"[CACHE] Using cached index data for {dataset_name}")
            return self.cache[dataset_name]

        print(f"[API] Loading index data for {dataset_name} via API...")
        try:
            tfidf_matrix = self.load_tfidf_matrix_via_api(dataset_name)
            doc_ids = self.load_doc_ids_via_api(f"{dataset_name}_all", vectorizer_type="tfidf")
            inverted_index = load_inverted_index(dataset_name)

            self.cache[dataset_name] = {
                'tfidf_matrix': tfidf_matrix,
                'doc_ids': doc_ids,
                'inverted_index': inverted_index
            }

            return self.cache[dataset_name]
        except Exception as e:
            print(f"Failed to load index data: {e}")
            raise


index_loader = IndexDataLoader()


def fetch_documents_text_via_api(dataset_name, doc_ids):
    """
    Fetch document texts from the FastAPI endpoint via HTTP GET request.

    Args:
        dataset_name (str): Dataset name.
        doc_ids (list): List of document IDs to fetch.

    Returns:
        dict: Mapping of doc_id to document text.
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
    """
    Match query against documents and rank them by similarity score.

    Args:
        query (str): Query text.
        dataset_name (str): Dataset name.
        similarity_threshold (float): Minimum similarity score to include.
        top_k (int or None): Limit number of top results.

    Returns:
        OrderedDict: Sorted mapping of doc_id to similarity score.
    """
    qp = QueryProcessor(dataset_name)
    query_vector, tokens = qp.process(query)

    # Load index data (cached or from API)
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
    documents_texts = fetch_documents_text_via_api(dataset_name, top_doc_ids)

    print(f"Top {top_k if top_k else 'all'} ranked results with document content:")
    for rank, (doc_id, score) in enumerate(sorted_ranking[:5], 1):
        print(f"🔹 Rank: {rank}")
        print(f"   Doc ID: {doc_id}")
        print(f"   Score: {score:.6f}")
        print("-" * 50)

    return OrderedDict(sorted_ranking)
