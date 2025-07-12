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


