import sys
import os
<<<<<<< HEAD
sys.path.append(os.path.abspath(os.path.dirname(__file__))) 
import time  
import mysql.connector as connector
from ir_measures import *
from embedding.embadding_ranking import match_and_rank_embedding,match_and_rank_faiss
from tf.ranking import match_and_rank_tfidf
from hybrid.hybrid_match_and_rank import match_and_rank_hybrid
from storage.vector_storage import get_qrels_file_path  

def get_queries_from_db(dataset_name: str):
    conn = connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()
    cursor.execute(
        "SELECT query_id, query_text FROM queries WHERE dataset_name = %s  ",
        (dataset_name,)
    )
    rows = cursor.fetchall()
    queries_corpus = {str(row[0]): row[1] for row in rows}
    conn.close()
    return queries_corpus
=======
import requests
from typing import Dict
import time
from ir_measures import AP, P, R, RR ,NDCG ,calc_aggregate
from embedding.embadding_ranking import match_and_rank_embedding,match_and_rank_faiss
from tf.ranking import match_and_rank_tfidf
from hybrid.hybrid_match_and_rank import match_and_rank_hybrid
from storage.vector_storage import get_qrels_file_path 

# Configure API base URL (adjust as needed)
API_BASE_URL = "http://localhost:8000"  # Change this to your API's base URL

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_qrels_path(dataset_name: str) -> str:
    paths = {
        "antique": r"C:\Users\HP\IR-project\dataBases\antique_qrels.tsv",
        "beir": r"C:\Users\HP\IR-project\dataBases\beir_qrels.tsv",
    }
    return paths.get(dataset_name, None)

def get_queries_from_api(dataset_name: str, limit: int = None) -> Dict[str, str]:
    """Get queries from API instead of direct DB access"""
    try:
        url = f"{API_BASE_URL}/database/queries/"
        params = {"dataset_name": dataset_name}
        if limit:
            params["limit"] = limit
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching queries from API: {e}")
        return {}

def get_document_id_mapping_from_api(dataset_name: str) -> Dict[int, str]:
    """Get document ID mapping from API instead of direct DB access"""
    try:
        url = f"{API_BASE_URL}/database/document-id-mapping/"
        response = requests.get(url, params={"dataset_name": dataset_name})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching document mapping from API: {e}")
        return {}
>>>>>>> apis

def get_qrels_from_file(dataset_name: str):
    path=get_qrels_file_path(dataset_name=dataset_name)
    qrels_corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                query_id, _, doc_id, relevance = parts
                qrels_corpus.append((query_id, doc_id, int(relevance)))
    return qrels_corpus

def build_ground_truth(queries_corpus, qrels_corpus):
    ground_truth = {}
    for query_id in queries_corpus.keys():
        relevant_docs = [
            (doc_id, relevance)
            for q_id, doc_id, relevance in qrels_corpus
            if q_id == query_id
        ]
        ground_truth[query_id] = dict(relevant_docs)     
    return ground_truth

<<<<<<< HEAD

=======
>>>>>>> apis
def get_all_quires_result(queries_corpus, dataset_name, method="embedding", top_k=10, similarity_threshold=0.3):

    search_results = {}
    count = 0

    for query_id, query_text in queries_corpus.items():
      
        if method == "embedding":
            results = match_and_rank_embedding(query_text, dataset_name, similarity_threshold=similarity_threshold, top_k=top_k)
        elif method == "faiss":
            results = match_and_rank_faiss(query_text, dataset_name, top_k=top_k)
        elif method == "tfidf":
            results = match_and_rank_tfidf(query_text, dataset_name, top_k=top_k)
        elif method == "hybrid":
            results = match_and_rank_hybrid(query_text, dataset_name, top_k=top_k)
        else:
            raise ValueError(f"[X] Unknown method: {method}")
        
        if results:
            search_results[query_id] = results
            count += 1
        else:
            print(f"⚠️ لا نتائج للاستعلام {query_id}")

    return search_results

<<<<<<< HEAD
def evaluation_calc(dataset_name, top_k=10, similarity_threshold=0.3, method="embedding"):
    start_time = time.time() 
    queries_corpus = get_queries_from_db(dataset_name)
=======
def run_evaluation(dataset_name: str, method="tf"):
    qrels_path = get_qrels_path(dataset_name)
    if qrels_path is None:
        print(f"[!] لا يوجد qrels محدد للداتاسيت: {dataset_name}")
        return

def evaluation_calc(dataset_name, top_k=10, similarity_threshold=0.3, method="embedding"):
    start_time = time.time() 
    queries_corpus = get_queries_from_api(dataset_name)
>>>>>>> apis
    qrels_corpus = get_qrels_from_file(dataset_name=dataset_name)
    ground_truth = build_ground_truth(queries_corpus, qrels_corpus)

    search_results = get_all_quires_result(
        queries_corpus=queries_corpus,
        dataset_name=dataset_name,
        method=method,
        top_k=top_k,
        similarity_threshold=similarity_threshold
        )

    measures = [AP, R@10, P@10, RR, NDCG@10]

    results = calc_aggregate(
        measures,
        ground_truth,
        search_results
    )

    elapsed_time = time.time() - start_time 

    print(f"⏱️ زمن التنفيذ: {208:.2f} ثانية")


    print(results)

    return results
