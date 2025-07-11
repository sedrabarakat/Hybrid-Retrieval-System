import sys
import os
from pathlib import Path # استيراد Path

# تصحيح حساب project_root
# هذا سيأخذك من C:\Users\HP\IR-project\query_processing\with_clustring\ranking_clustring.py
# إلى C:\Users\HP\IR-project\
# تم تغيير parents[3] إلى parents[2]
project_root = Path(__file__).resolve().parents[2] 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import time # تم استيراد وحدة الوقت

# استيراد load_clusters هنا
from storage.vector_storage import load_tfidf_matrix, load_doc_ids, load_clusters 
from indexing.inverted_index_loader import load_inverted_index
from tf.query_processing import QueryProcessor

import mysql.connector

# تصحيح حساب project_root
project_root = Path(__file__).resolve().parents[2] 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# الحل لمشكلة 'No module named 'TF_IDF''
import vectorize.tokenizer_definition 
sys.modules["TF_IDF"] = vectorize.tokenizer_definition


class IndexDataLoader:
    _cache = {} # استخدام كاش على مستوى الفئة

    def __init__(self):
        pass # لا حاجة لتهيئة هنا

    def load(self, dataset_name):
        if dataset_name in self._cache:
            print(f"[CACHE] Using cached index data for {dataset_name}")
            return self._cache[dataset_name]

        print(f"[DISK] Loading index data for {dataset_name} from disk...")
        tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
        doc_ids = load_doc_ids(f"{dataset_name}_all")
        inverted_index = load_inverted_index(dataset_name)
        
        # **تحميل معلومات المجموعات هنا**
        clusters_data = load_clusters(f"{dataset_name}_all") 

        # **نقل إنشاء doc_id_to_index إلى هنا وتخزينه مؤقتًا**
        print(f"Creating doc_id_to_index map for {len(doc_ids)} documents...")
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        print("Doc_id_to_index map created.")

        self._cache[dataset_name] = {
            'tfidf_matrix': tfidf_matrix,
            'doc_ids': doc_ids,
            'inverted_index': inverted_index,
            'clusters': clusters_data, # إضافة بيانات المجموعات إلى الكاش
            'doc_id_to_index': doc_id_to_index # إضافة doc_id_to_index إلى الكاش
        }
        print(f"[DISK] Index data for {dataset_name} loaded and cached.")
        return self._cache[dataset_name]


index_loader = IndexDataLoader() # أنشئ loader مرة واحدة (global)


def load_documents_text(dataset_name, doc_ids, batch_size=1000):
    """
    يحمل نصوص المستندات من قاعدة البيانات بناءً على معرّفاتها.
    يستخدم الجلب على دفعات لتحسين الأداء.
    """
    if not doc_ids:
        return {}

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor(dictionary=True)

    all_rows = []
    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i:i+batch_size]
        placeholders = ','.join(['%s'] * len(batch_ids))
        query = f"""
            SELECT id, text
            FROM documents
            WHERE id IN ({placeholders})
              AND dataset_name = %s
        """
        cursor.execute(query, (*batch_ids, dataset_name))
        rows = cursor.fetchall()
        all_rows.extend(rows)

    conn.close()
    return {row['id']: row['text'] for row in all_rows}


def match_and_rank(query: str, dataset_name: str, similarity_threshold=0.0001, top_k=None, use_clusters: bool = False):
    """
    يقوم بمطابقة وترتيب المستندات لاستعلام معين، مع إمكانية استخدام تصفية المجموعات.

    Parameters:
    - query (str): نص الاستعلام.
    - dataset_name (str): اسم مجموعة البيانات.
    - similarity_threshold (float): الحد الأدنى لدرجة التشابه لتضمين المستند.
    - top_k (int, optional): عدد أفضل المستندات المراد إرجاعها.
    - use_clusters (bool): إذا كانت True، سيتم تصفية المستندات بناءً على المجموعات ذات الصلة بالاستعلام.
    """
    start_time_total = time.time() 

    print(f"Starting match and rank for query: '{query}' on dataset: '{dataset_name}' (Clusters enabled: {use_clusters})")
    
    qp = QueryProcessor(dataset_name)
    query_vector, tokens = qp.process(query)
    print(f"Query processed. Tokens: {tokens}")

    data = index_loader.load(dataset_name)
    tfidf_matrix = data['tfidf_matrix']
    doc_ids = data['doc_ids']
    inverted_index = data['inverted_index']
    clusters_data = data['clusters'] # تحميل بيانات المجموعات
    doc_id_to_index = data['doc_id_to_index'] # الحصول على doc_id_to_index من الكاش

    matched_tokens = [t for t in tokens if t in inverted_index]
    if not matched_tokens:
        print("[!] لم تُوجد كلمات من الاستعلام في الفهرس المعكوس. إرجاع نتيجة فارغة.")
        end_time_total = time.time()
        print(f"Total match and rank execution time: {end_time_total - start_time_total:.2f} seconds.")
        return OrderedDict()

    candidate_doc_ids = set()
    for token in matched_tokens:
        candidate_doc_ids.update(inverted_index[token])

    # **تطبيق منطق تصفية المجموعات بناءً على use_clusters**
    filtered_candidate_indices = []
    if use_clusters:
        print("Determining relevant clusters for the query...")
        cluster_relevance_counts = np.zeros(clusters_data.max() + 1)
        for token in matched_tokens:
            if token in inverted_index:
                doc_indices_for_token = [doc_id_to_index[d_id] for d_id in inverted_index[token] if d_id in doc_id_to_index]
                for doc_idx in doc_indices_for_token:
                    cluster_id = clusters_data[doc_idx]
                    cluster_relevance_counts[cluster_id] += 1
        
        if cluster_relevance_counts.sum() == 0:
            print("[!] لا توجد مجموعات ذات صلة بالاستعلام بناءً على الكلمات المطابقة. البحث في جميع المستندات.")
            relevant_clusters = None 
        else:
            most_relevant_cluster_id = np.argmax(cluster_relevance_counts)
            relevant_clusters = [most_relevant_cluster_id] 
            print(f"Query most relevant to cluster(s): {relevant_clusters}")

        print(f"Filtering candidate documents to relevant clusters: {relevant_clusters}")
        if relevant_clusters is not None:
            for doc_id in candidate_doc_ids:
                if doc_id in doc_id_to_index:
                    doc_idx = doc_id_to_index[doc_id]
                    if clusters_data[doc_idx] in relevant_clusters:
                        filtered_candidate_indices.append(doc_idx)
        else: # إذا لم يتم تحديد مجموعات ذات صلة، استخدم جميع المرشحين
            filtered_candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]
    else: # إذا كانت use_clusters False، استخدم جميع المرشحين من الفهرس المعكوس
        print("Clusters disabled. Searching across all candidate documents.")
        filtered_candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]


    if not filtered_candidate_indices:
        print("[!] لا توجد مؤشرات لوثائق مطابقة بعد التصفية بالمجموعات. إرجاع نتيجة فارغة.")
        end_time_total = time.time()
        print(f"Total match and rank execution time: {end_time_total - start_time_total:.2f} seconds.")
        return OrderedDict()

    filtered_candidate_indices = sorted(filtered_candidate_indices)
    candidate_vectors = tfidf_matrix[filtered_candidate_indices]

    print(f"Calculating cosine similarity for {len(filtered_candidate_indices)} candidate documents...")
    similarity_scores = cosine_similarity(query_vector, candidate_vectors).flatten()

    ranking = {
        doc_ids[i]: float(score)
        for i, score in zip(filtered_candidate_indices, similarity_scores)
        if score >= similarity_threshold
    }

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_ranking = sorted_ranking[:top_k]

    top_doc_ids = [doc_id for doc_id, _ in sorted_ranking]
    print(f"Loading text for top {len(top_doc_ids)} documents...") 
    documents_texts = load_documents_text(dataset_name, top_doc_ids) 

    print(f"[14] عرض أعلى {top_k if top_k else 'الكل'} نتائج مرتبة مع محتوى الوثائق:")
    for rank, (doc_id, score) in enumerate(sorted_ranking[:5], 1):
        text = documents_texts.get(doc_id, "[النص غير موجود]") 
        print(f"🔹 Rank: {rank}")
        print(f"   Doc ID: {doc_id}")
        print(f"   Score: {score:.6f}")
        print(f"   Text: {text[:200]}...") 
        print("-" * 50)

    end_time_total = time.time()
    print(f"Match and rank completed. Found {len(sorted_ranking)} relevant documents.")
    print(f"Total match and rank execution time: {end_time_total - start_time_total:.2f} seconds.")
    return OrderedDict(sorted_ranking)

# مثال على كيفية الاستخدام (في ملفك الرئيسي أو ملف التشغيل):
# match_and_rank(query="What is the capital of France?", dataset_name="beir", top_k=10, use_clusters=True)
