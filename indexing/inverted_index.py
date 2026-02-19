<<<<<<< HEAD
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import joblib
from collections import defaultdict
from storage.vector_storage import (
    load_vectorizer,
    load_tfidf_matrix,
    load_tfidf_ids,
)

def build_inverted_index_tfidf(dataset_name: str):
    # تحميل الـ vectorizer
    vectorizer = load_vectorizer(dataset_name)

    # تحميل المصفوفة
    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")

    # تحميل doc_ids بالترتيب الصحيح
    doc_ids = load_tfidf_ids(dataset_name)

    # جلب الـ terms (الكلمات) من الـ vectorizer
    terms = vectorizer.get_feature_names_out()

    inverted_index = defaultdict(list)

    # نمرّ على كل كلمة
    for term_idx, term in enumerate(terms):
        # نحصل على العمود الخاص بالكلمة
        column = tfidf_matrix[:, term_idx]
        # نجيب إندكسات الصفوف (أي المستندات التي تحتوي الكلمة)
        nonzero_rows = column.nonzero()[0]
        for row_idx in nonzero_rows:
            tfidf_value = column[row_idx, 0]
            real_doc_id = doc_ids[row_idx]
            inverted_index[term].append((real_doc_id, float(tfidf_value)))

    # حفظ الفهرس بنفس مجلد الملف
    output_folder = os.path.dirname(__file__)
    index_path = os.path.join(output_folder, f"{dataset_name}_inverted_index.joblib")

    joblib.dump(dict(inverted_index), index_path, compress=3)


    print(f"[✓] تم بناء وحفظ الفهرس المعكوس بـ TF-IDF لمجموعة البيانات: {dataset_name}")
    # طباعة أول 5 كلمات كعينات
    for term, postings in list(inverted_index.items())[:5]:
        print(f"Term: {term}")
        for doc_id, score in postings:
            print(f"  Doc ID: {doc_id} → TF-IDF: {score:.4f}")
        print("-" * 40)


def test_single_token_inverted_index(dataset_name: str):
    # تحميل vectorizer
    vectorizer = load_vectorizer(dataset_name)
    # تحميل المصفوفة
    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    # تحميل doc_ids بنفس الترتيب
    doc_ids = load_tfidf_ids(dataset_name)
    # طباعة عدد المستندات وعدد التوكينات
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of documents: {len(doc_ids)}")
    print(f"Number of tokens: {len(vectorizer.get_feature_names_out())}")
    print("-" * 60)
    # نختار أول توكين فقط للتجربة
    terms = vectorizer.get_feature_names_out()
    target_term = "iraq"
    term_idx = vectorizer.vocabulary_.get(target_term)
    if term_idx is None:
        print(f"[!] التوكين '{target_term}' غير موجود في vectorizer.")
        return
    first_term = target_term
    column = tfidf_matrix[:, term_idx]  
    print(f"Chosen test term: {first_term}")
    # نحصل على العمود الخاص بالتوكين
    term_idx = 0
    column = tfidf_matrix[:, term_idx]
    # نبني الفهرس لهذا التوكين فقط
    inverted_index = []
    for row_idx in column.nonzero()[0]:
        tfidf_value = column[row_idx, 0]
        real_doc_id = doc_ids[row_idx]
        inverted_index.append((real_doc_id, float(tfidf_value)))
    # طباعة الفهرس للتوكين المختار
    print(f"\nInverted index for token '{first_term}':")
    for doc_id, tfidf in inverted_index:
        print(f"Doc ID: {doc_id} → TF-IDF: {tfidf:.6f}")


=======
from fastapi import APIRouter, HTTPException
from .text_preprocessing import get_preprocessed_text_terms
import requests
import sys
import os
from typing import List, Dict, Any
from collections import defaultdict
import joblib

router = APIRouter()

@router.get("/text_processing/get_preprocessed_text_terms")
def get_preprocessed_text_terms_request(text: str, dataset_name: str):
    processed_tokens = get_preprocessed_text_terms(text, dataset_name)
    return processed_tokens

# ====================== DOCUMENT PROCESSING ======================
def fetch_documents(dataset_name: str, skip: int, limit: int) -> List[Dict[str, Any]]:
    """Fetch a batch of documents from the API."""
    try:
        url = f"http://localhost:8000/database/text?dataset_name={dataset_name}&skip={skip}&limit={limit}"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching documents: {str(e)}")
        return []

def fetch_all_documents(dataset_name: str, batch_size: int = 1000) -> List[Dict[str, Any]]:
    """Fetch all documents in batches."""
    all_docs = []
    skip = 0
    
    while True:
        batch = fetch_documents(dataset_name, skip, batch_size)
        if not batch:
            break
        all_docs.extend(batch)
        print(f"Fetched batch {skip//batch_size + 1} ({len(batch)} documents)")
        skip += batch_size
    
    return all_docs

def call_preprocessing_api(text: str, dataset_name: str) -> List[str]:
    """Call the preprocessing API endpoint."""
    try:
        url = f"http://localhost:8000/text_processing/get_preprocessed_text_terms?text={text}&dataset_name={dataset_name}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling preprocessing API: {str(e)}")
        return []

def build_inverted_index(dataset_name: str):
    rows = fetch_all_documents(dataset_name)

    inverted_index = defaultdict(set)

    for doc_id, raw_text in rows:
        if not raw_text or not raw_text.strip():
            print(f"[!] Document {doc_id} skipped (empty raw text).")
            continue

        terms = call_preprocessing_api(raw_text, dataset_name)

        if not terms:
            print(f"[!] Document {doc_id} skipped (no terms after processing).")
            continue

        for term in set(terms):
            inverted_index[term].add(doc_id)

    inverted_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}

    output_folder = r"C:\Users\HP\IR-project\indexing\saved_models\inverted_index"
    os.makedirs(output_folder, exist_ok=True)

    index_path = os.path.join(output_folder, f"{dataset_name}_inverted_index.joblib")
    joblib.dump(inverted_index, index_path, compress=3)

    print(f"[✓] Inverted index built and saved as joblib for: {dataset_name}")
>>>>>>> apis
