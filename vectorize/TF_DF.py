import sys
import os
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing.text_preprocessing import get_preprocessed_text_terms
import storage.vector_storage as storage

def tokenizer(text, dataset_name):
    return get_preprocessed_text_terms(text, dataset_name)

def build_save_vectorizer(dataset_name: str):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM documents WHERE dataset_name = %s LIMIT 1", (dataset_name,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        print(f"No document found for dataset '{dataset_name}'")
        return

    doc_id, raw_text = row
    raw_texts = [raw_text]


    # نمرر tokenizer مع dataset_name باستخدام partial
    tokenizer_with_dataset = partial(tokenizer, dataset_name=dataset_name)

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer_with_dataset,
        lowercase=False,
        preprocessor=None,
        token_pattern=None
    )

    tfidf_matrix = vectorizer.fit_transform(raw_texts)

    # حفظ vectorizer - الآن سيتم حفظه بدون مشاكل pickling
    file_suffix = f"doc_{doc_id}"
    storage.save_vectorizer(vectorizer, f"{dataset_name}_{file_suffix}")
    storage.save_tfidf_matrix(tfidf_matrix, f"{dataset_name}_{file_suffix}")


    embeddings = tfidf_matrix.toarray()
    print(f"[✓]  '{dataset_name}' :")
    print(embeddings[0])
