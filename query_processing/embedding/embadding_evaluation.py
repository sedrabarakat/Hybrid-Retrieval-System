# evaluation_emb.py

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import json
from typing import Dict
import mysql.connector
import ir_measures
from ir_measures import AP, P, R, RR

from embadding_ranking import match_and_rank_embedding


def get_qrels_path(dataset_name: str) -> str:
    paths = {
        "antique": r"C:\Users\HP\IR-project\dataBases\antique_qrels.tsv",
        "beir": r"C:\Users\HP\IR-project\dataBases\beir_qrels.tsv",
    }
    return paths.get(dataset_name, None)


def get_queries_from_db(dataset_name: str):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()
    cursor.execute(
        "SELECT query_id, query_text FROM queries WHERE dataset_name = %s",
        (dataset_name,)
    )
    rows = cursor.fetchall()
    queries_corpus = {str(row[0]): row[1] for row in rows}
    conn.close()
    return queries_corpus


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


def load_doc_id_mapping(dataset_name):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, document_id
        FROM documents
        WHERE dataset_name = %s
    """, (dataset_name,))
    rows = cursor.fetchall()
    conn.close()

    mapping = {int(row['id']): str(row['document_id']) for row in rows}
    return mapping


def get_search_results(dataset_name: str):
    search_results = {}
    queries_corpus = get_queries_from_db(dataset_name)
    doc_id_mapping = load_doc_id_mapping(dataset_name)

    for query_id, query in queries_corpus.items():
        print(f"Evaluating query {query_id}")
        results = match_and_rank_embedding(query, dataset_name, similarity_threshold=0.3)

        converted_results = {}
        for doc_pk, score in results.items():
            doc_real_id = doc_id_mapping.get(int(doc_pk))
            if doc_real_id:
                converted_results[doc_real_id] = score
            else:
                print(f"[!] مفقود mapping لـ doc_id: {doc_pk}")

        search_results[query_id] = converted_results

    return search_results


def run_evaluation(dataset_name: str):
    qrels_path = get_qrels_path(dataset_name)
    if qrels_path is None:
        print(f"[!] لا يوجد qrels محدد للداتاسيت: {dataset_name}")
        return

    queries_corpus = get_queries_from_db(dataset_name)
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

    print("=== EVALUATION RESULTS (Embedding) ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # حفظ النتائج إلى ملف JSON
    output_path = f"evaluation_results_embedding_{dataset_name}.json"

    results_json = {str(metric): value for metric, value in results.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    print(f"✅ تم حفظ نتائج التقييم في الملف: {output_path}")
