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
import mysql.connector

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

def load_documents_text(dataset_name, doc_ids):
   
    if not doc_ids:
        return {}

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor(dictionary=True)

    placeholders = ','.join(['%s'] * len(doc_ids))
    query = f"""
        SELECT id, text
        FROM documents
        WHERE id IN ({placeholders}) AND dataset_name = %s
    """
    cursor.execute(query, (*doc_ids, dataset_name))
    rows = cursor.fetchall()

    conn.close()

    return {row['id']: row['text'] for row in rows}


# أنشئ loader مرة واحدة (global)

def match_and_rank(query: str, dataset_name: str, similarity_threshold=0.0001, top_k=None):
    qp = QueryProcessor(dataset_name)
    query_vector, tokens = qp.process(query)

    # استخدم الكاش هنا
    data = index_loader.load(dataset_name)

    tfidf_matrix = data['tfidf_matrix']
    doc_ids = data['doc_ids']
    inverted_index = data['inverted_index']

 

    matched_tokens = [t for t in tokens if t in inverted_index]

    if not matched_tokens:
        print("[!] لم تُوجد كلمات من الاستعلام في الفهرس المعكوس. إرجاع نتيجة فارغة.")
        return OrderedDict()

    candidate_doc_ids = set()
    for token in matched_tokens:
        candidate_doc_ids.update(inverted_index[token])

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    missing_doc_ids = [doc_id for doc_id in candidate_doc_ids if doc_id not in doc_id_to_index]
    if missing_doc_ids:
        print("🔍 عينة منها:", missing_doc_ids[:10])

    candidate_indices = [
        doc_id_to_index[doc_id] for doc_id in candidate_doc_ids
        if doc_id in doc_id_to_index
    ]

    if not candidate_indices:
        print("[!] لا توجد مؤشرات لوثائق مطابقة. إرجاع نتيجة فارغة.")
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

    # تحميل نصوص المستندات من قاعدة البيانات
    top_doc_ids = [doc_id for doc_id, _ in sorted_ranking]
    documents_texts = load_documents_text(dataset_name, top_doc_ids)

    print(f"[14] عرض أعلى {top_k if top_k else 'الكل'} نتائج مرتبة مع محتوى الوثائق:")
    for rank, (doc_id, score) in enumerate(sorted_ranking[:5], 1):
        text = documents_texts.get(doc_id, "[النص غير موجود]")
        print(f"🔹 Rank: {rank}")
        print(f"   Doc ID: {doc_id}")
        print(f"   Score: {score:.6f}")
        print(f"   Text: {text[:200]}...")
        print("-" * 50)

    return OrderedDict(sorted_ranking)

   