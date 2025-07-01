from pathlib import Path
import sys
import os
import requests
from typing import Dict
import time
import json
import ir_measures
from ir_measures import AP, P, R, RR
from tf.ranking import match_and_rank
from embedding.embadding_ranking import match_and_rank_embedding

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

def get_qrels_from_file(path: str):
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

def get_search_results(dataset_name: str, method="tf"):
    search_results = {}
    queries_corpus = get_queries_from_api(dataset_name)
    qrels_corpus = get_qrels_from_file(get_qrels_path(dataset_name))
    qrels_query_ids = set(q_id for q_id, _, _ in qrels_corpus)

    doc_id_mapping = get_document_id_mapping_from_api(dataset_name)

    for query_id, query_text in queries_corpus.items():
        if query_id not in qrels_query_ids:
            continue

        print(f"Evaluating query {query_id}")

        if method == "tf":
            results = match_and_rank(query_text, dataset_name)
        elif method == "embedding":
            results = match_and_rank_embedding(query_text, dataset_name, similarity_threshold=0.3)
        else:
            raise ValueError(f"Unsupported method: {method}")

        converted_results = {}
        for doc_pk, score in results.items():
            doc_real_id = doc_id_mapping.get(int(doc_pk))
            if doc_real_id:
                converted_results[doc_real_id] = score
            else:
                print(f"[!] مفقود mapping لـ doc_id: {doc_pk}")

        search_results[query_id] = converted_results

    return search_results

def run_evaluation(dataset_name: str, method="tf"):
    start_time = time.time()  # ⏱️ بداية التوقيت

    qrels_path = get_qrels_path(dataset_name)
    if qrels_path is None:
        print(f"[!] لا يوجد qrels محدد للداتاسيت: {dataset_name}")
        return

    queries_corpus = get_queries_from_api(dataset_name)
    qrels_corpus = get_qrels_from_file(qrels_path)

    ground_truth = build_ground_truth(queries_corpus, qrels_corpus)
    search_results = get_search_results(dataset_name, method=method)

    measures = [AP, R@10, P@10, RR]

    results = ir_measures.calc_aggregate(
        measures,
        ground_truth,
        search_results
    )

    print("=== EVALUATION RESULTS ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # ⏱️ حساب الزمن المستغرق
    elapsed_time = time.time() - start_time
    print(f"⏱️ زمن التنفيذ: {elapsed_time:.2f} ثانية")

    # تخزين النتائج مع زمن التنفيذ
    output_path = f"evaluation_results_{method}_{dataset_name}.json"
    results_json = {
        str(metric): value for metric, value in results.items()
    }
    results_json["execution_time_sec"] = round(elapsed_time, 2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    print(f"✅ تم حفظ نتائج التقييم في الملف: {output_path}")