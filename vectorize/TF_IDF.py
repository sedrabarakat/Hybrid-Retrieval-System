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
    cursor.execute("SELECT id, text FROM documents WHERE dataset_name = %s", (dataset_name,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print(f"No documents found for dataset '{dataset_name}'")
        return

    for doc_id, raw_text in rows:
        tokens = tokenizer(raw_text, dataset_name)

        if not tokens:
            print(f"[!] المستند {doc_id} تم حذفه لأنه فارغ بعد التنظيف.")
            continue

        tokenizer_with_dataset = partial(tokenizer, dataset_name=dataset_name)

        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer_with_dataset,
            lowercase=False,
            preprocessor=None,
            token_pattern=None
        )

        tfidf_matrix = vectorizer.fit_transform([raw_text])

        file_suffix = f"doc_{doc_id}"
        vectorizer_type = "tfidf"

        storage.save_vectorizer(vectorizer, f"{dataset_name}_{file_suffix}", vectorizer_type=vectorizer_type)
        storage.save_tfidf_matrix(tfidf_matrix, f"{dataset_name}_{file_suffix}", vectorizer_type=vectorizer_type)

        embeddings = tfidf_matrix.toarray()
        print(f"[✓] Dataset: '{dataset_name}', Doc ID: {doc_id}")
        print(embeddings[0])
