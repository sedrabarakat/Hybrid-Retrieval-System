from pathlib import Path
import sys
import requests
import json
import ir_measures
from ir_measures import AP, P, R, RR
from ranking import match_and_rank
from typing import Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parents[2]))

# API base URL (update this to match your FastAPI server address)
API_BASE_URL = "http://localhost:8000"

def get_qrels_path(dataset_name: str) -> str:
    """
    Returns the path to qrels file based on dataset name.
    Modify these paths if you add new datasets.
    """
    paths = {
        "antique": r"C:\Users\HP\IR-project\dataBases\antique_qrels.tsv",
        "beir": r"C:\Users\HP\IR-project\dataBases\beir_qrels.tsv",
    }
    return paths.get(dataset_name, None)

def get_queries_from_api(dataset_name: str) -> Dict[str, str]:
    """Get queries from FastAPI endpoint"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/database/queries/?dataset_name={dataset_name}&limit=3"
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching queries: {e}")
        return {}

def get_doc_id_mapping_from_api(dataset_name: str) -> Dict[int, str]:
    """Get document ID mapping from FastAPI endpoint"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/database/document-id-mapping?dataset_name={dataset_name}"
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching document ID mapping: {e}")
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

def build_ground_truth(queries_corpus: Dict[str, str], qrels_corpus: list):
    ground_truth = {}
    for query_id in queries_corpus.keys():
        relevant_docs = [(doc_id, relevance) 
                         for q_id, doc_id, relevance in qrels_corpus 
                         if q_id == query_id]
        ground_truth[query_id] = dict(relevant_docs)
    return ground_truth

def get_search_results(dataset_name: str) -> Dict[str, Dict[str, float]]:
    search_results = {}
    queries_corpus = get_queries_from_api(dataset_name)
    doc_id_mapping = get_doc_id_mapping_from_api(dataset_name)

    for query_id, query in queries_corpus.items():
        print(f'Evaluating query {query_id}')
        results = match_and_rank(query, dataset_name)

        converted_results = {}
        for doc_pk, score in results.items():
            doc_real_id = doc_id_mapping.get(int(doc_pk))
            if doc_real_id:
                converted_results[doc_real_id] = score
            else:
                print(f"[!] Missing mapping for doc_id: {doc_pk}")

        search_results[query_id] = converted_results

    return search_results

def run_evaluation(dataset_name: str):
    qrels_path = get_qrels_path(dataset_name)
    if qrels_path is None:
        print(f"[!] No qrels defined for dataset: {dataset_name}")
        return

    queries_corpus = get_queries_from_api(dataset_name)
    qrels_corpus = get_qrels_from_file(qrels_path)

    ground_truth = build_ground_truth(queries_corpus, qrels_corpus)
    search_results = get_search_results(dataset_name)

    measures = [
        AP,
        R@10,
        P@10,
        RR
    ]

    results = ir_measures.calc_aggregate(
        measures,
        ground_truth,
        search_results
    )

    print("=== EVALUATION RESULTS ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # Save results to JSON file
    output_path = f"evaluation_results_{dataset_name}.json"
    results_json = {str(metric): value for metric, value in results.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    print(f"✅ Evaluation results saved to: {output_path}")

if __name__ == "__main__":
    dataset_name = "antique"  # Change this to your dataset name
    run_evaluation(dataset_name)