import sys
import os
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

def evaluation_calc(dataset_name, top_k=10, similarity_threshold=0.3, method="embedding"):
    start_time = time.time() 
    queries_corpus = get_queries_from_db(dataset_name)
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
