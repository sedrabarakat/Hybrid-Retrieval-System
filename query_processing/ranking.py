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
    print("[1] بدء معالجة الاستعلام...")
    qp = QueryProcessor(dataset_name)
    query_vector, tokens = qp.process(query)
    print(f"[2] تم استخراج التوكنز من الاستعلام: {tokens}")

    print("[3] تحميل تمثيلات TF-IDF...")
    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    print(f"[4] شكل مصفوفة TF-IDF: {tfidf_matrix.shape}")

    print("[5] تحميل قائمة معرفات الوثائق...")
    doc_ids = load_doc_ids(f"{dataset_name}_all")
    print(f"[6] عينة من doc_ids: {doc_ids[:5]} (كلها من نوع: {type(doc_ids[0])})")

    print("[7] تحميل الفهرس المعكوس...")
    inverted_index = load_inverted_index(dataset_name)
    print(f"[8] عدد المفاتيح في الفهرس المعكوس: {len(inverted_index)}")

    matched_tokens = [t for t in tokens if t in inverted_index]
    print(f"[9] التوكنز الموجودة في الفهرس المعكوس: {matched_tokens}")

    if not matched_tokens:
        print("[!] لم تُوجد كلمات من الاستعلام في الفهرس المعكوس. إرجاع نتيجة فارغة.")
        return OrderedDict()

    candidate_doc_ids = set()
    for token in matched_tokens:
        candidate_doc_ids.update(inverted_index[token])
    print(f"[10] عدد الوثائق المرشحة بعد الفحص: {len(candidate_doc_ids)}")

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    print(f"[11] تم بناء قاموس للمطابقة بين doc_id و index.")

    print("🔍 عينة من candidate_doc_ids (فهرس معكوس):", list(candidate_doc_ids)[:10])
    print("🔍 عينة من doc_ids المحملة:", doc_ids[:10])

    missing_doc_ids = [doc_id for doc_id in candidate_doc_ids if doc_id not in doc_id_to_index]
    print(f"🔍 عدد doc_ids في الفهرس المعكوس غير موجودة في doc_id_to_index: {len(missing_doc_ids)}")
    print("🔍 عينة منها:", missing_doc_ids[:10])

    candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]
    print(f"[12] عدد مؤشرات الوثائق المرشحة بعد المطابقة: {len(candidate_indices)}")

    if not candidate_indices:
        print("[!] لا توجد مؤشرات لوثائق مطابقة. إرجاع نتيجة فارغة.")
        return OrderedDict()

    candidate_indices = sorted(candidate_indices)
    candidate_vectors = tfidf_matrix[candidate_indices]
    print(f"[13] تم استخراج تمثيلات TF-IDF للوثائق المرشحة. الشكل: {candidate_vectors.shape}")

    similarity_scores = cosine_similarity(query_vector, candidate_vectors).flatten()
    print(f"[14] تم حساب درجات التشابه. عدد الدرجات: {len(similarity_scores)}")

    ranking = {
        doc_ids[i]: float(score)
        for i, score in zip(candidate_indices, similarity_scores)
        if score >= similarity_threshold
    }
    print(f"[15] عدد الوثائق بعد تطبيق عتبة التشابه ({similarity_threshold}): {len(ranking)}")

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_ranking = sorted_ranking[:top_k]
    print(f"[16] عرض أعلى {top_k if top_k else 'الكل'} نتائج مرتبة:")

    for rank, (doc_id, score) in enumerate(sorted_ranking[:5], 1):
        print(f"    {rank}. Doc ID: {doc_id}, Score: {score:.6f}")

    return OrderedDict(sorted_ranking)
    print("[1] بدء معالجة الاستعلام...")
    qp = QueryProcessor(dataset_name)
    query_vector, tokens = qp.process(query)
    print(f"[2] تم استخراج التوكنز من الاستعلام: {tokens}")

    print("[3] تحميل تمثيلات TF-IDF...")
    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    print(f"[4] شكل مصفوفة TF-IDF: {tfidf_matrix.shape}")

    print("[5] تحميل قائمة معرفات الوثائق...")
    doc_ids = load_doc_ids(f"{dataset_name}_all")
    print(f"[6] عينة من doc_ids: {doc_ids[:5]} (كلها من نوع: {type(doc_ids[0])})")

    print("[7] تحميل الفهرس المعكوس...")
    inverted_index = load_inverted_index(dataset_name)
    print(f"[8] عدد المفاتيح في الفهرس المعكوس: {len(inverted_index)}")

    matched_tokens = [t for t in tokens if t in inverted_index]
    print(f"[9] التوكنز الموجودة في الفهرس المعكوس: {matched_tokens}")

    if not matched_tokens:
        print("[!] لم تُوجد كلمات من الاستعلام في الفهرس المعكوس. إرجاع نتيجة فارغة.")
        return OrderedDict()

    candidate_doc_ids = set()
    for token in matched_tokens:
        candidate_doc_ids.update(inverted_index[token])
    print(f"[10] عدد الوثائق المرشحة بعد الفحص: {len(candidate_doc_ids)}")

    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    print(f"[11] تم بناء قاموس للمطابقة بين doc_id و index.")

    candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]
    print(f"[12] عدد مؤشرات الوثائق المرشحة بعد المطابقة: {len(candidate_indices)}")

    if not candidate_indices:
        print("[!] لا توجد مؤشرات لوثائق مطابقة. إرجاع نتيجة فارغة.")
        return OrderedDict()

    candidate_indices = sorted(candidate_indices)
    candidate_vectors = tfidf_matrix[candidate_indices]
    print(f"[13] تم استخراج تمثيلات TF-IDF للوثائق المرشحة. الشكل: {candidate_vectors.shape}")

    similarity_scores = cosine_similarity(query_vector, candidate_vectors).flatten()
    print(f"[14] تم حساب درجات التشابه. عدد الدرجات: {len(similarity_scores)}")

    ranking = {
        doc_ids[i]: float(score)
        for i, score in zip(candidate_indices, similarity_scores)
        if score >= similarity_threshold
    }
    print(f"[15] عدد الوثائق بعد تطبيق عتبة التشابه ({similarity_threshold}): {len(ranking)}")

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_ranking = sorted_ranking[:top_k]
    print(f"[16] عرض أعلى {top_k if top_k else 'الكل'} نتائج مرتبة:")

    for rank, (doc_id, score) in enumerate(sorted_ranking[:5], 1):
        print(f"    {rank}. Doc ID: {doc_id}, Score: {score:.6f}")

    return OrderedDict(sorted_ranking)
    qp = QueryProcessor(dataset_name)
    query_vector, tokens = qp.process(query)

    print("✅ Step 1 - Tokens after preprocessing:", tokens)

    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    doc_ids = load_doc_ids(f"{dataset_name}_all")
    inverted_index = load_inverted_index(dataset_name)

    print("✅ Step 2 - Checking which tokens exist in the inverted index:")
    matched_tokens = [t for t in tokens if t in inverted_index]
    print("✅ Tokens found in index:", matched_tokens)

    if not matched_tokens:
        print("⚠️ لا توجد أي كلمات من الاستعلام موجودة في الفهرس.")
        return OrderedDict()

    # تحويل doc_id إلى index
    doc_id_to_index = {str(doc_id): idx for idx, doc_id in enumerate(doc_ids)}
    print("✅ Tokens found in index:", doc_id_to_index)

    candidate_doc_indices = set()
    for token in matched_tokens:
        for doc_id in inverted_index[token]:
            index = doc_id_to_index.get(doc_id)
            if index is not None:
                candidate_doc_indices.add(index)

    print("✅ Step 3 - Number of candidate documents found:", len(candidate_doc_indices))

    if not candidate_doc_indices:
        print("⚠️ لم يتم العثور على أي وثائق مرشحة.")
        return OrderedDict()

    candidate_doc_indices = sorted(candidate_doc_indices)
    candidate_doc_vectors = tfidf_matrix[candidate_doc_indices]

    print("✅ Step 4 - Calculating cosine similarity...")
    similarity_scores = cosine_similarity(query_vector, candidate_doc_vectors).flatten()

    print("✅ Step 5 - Building ranked result...")
    ranking = {
        doc_ids[i]: float(score)
        for i, score in zip(candidate_doc_indices, similarity_scores)
        if score >= similarity_threshold
    }

    print("✅ Step 6 - Number of documents above threshold:", len(ranking))

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_ranking = sorted_ranking[:top_k]

    print("✅ Step 7 - Top results preview:", sorted_ranking[:5])

    return OrderedDict(sorted_ranking)