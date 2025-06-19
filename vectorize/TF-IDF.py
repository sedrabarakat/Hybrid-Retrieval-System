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

def build_save_vectorizer_first_doc_only(dataset_name: str):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT processed_text FROM documents WHERE dataset_name = %s LIMIT 1", (dataset_name,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        print(f"No document found for dataset '{dataset_name}'")
        return

    raw_texts = [row[0]]

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
    storage.save_vectorizer(vectorizer, dataset_name + "_first_doc")
    storage.save_tfidf_matrix(tfidf_matrix, dataset_name + "_first_doc")

    embeddings = tfidf_matrix.toarray()
    print(f"[✓] First embedding for '{dataset_name}' (first doc only):")
    print(embeddings[0])

if __name__ == "__main__":
    datasets = ["beir", "antique", "quora"]
    for ds in datasets:
        build_save_vectorizer_first_doc_only(ds)
