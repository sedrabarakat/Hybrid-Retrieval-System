import nltk
nltk.data.path.append("C:/Users/Barakat/AppData/Roaming/nltk_data")
import sys
import os
from functools import partial
import vectorize.tokenizer_definition




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
    cursor.execute("SELECT id, text FROM documents WHERE dataset_name = %s ", (dataset_name,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print(f"[!] لا يوجد مستندات في مجموعة البيانات: '{dataset_name}'")
        return

    tokenizer_with_dataset = partial(tokenizer, dataset_name=dataset_name)
    raw_texts = []
    doc_ids = []

    for doc_id, raw_text in rows: 
        raw_texts.append(raw_text)
        doc_ids.append(doc_id)

    vectorizer = TfidfVectorizer(
            tokenizer=tokenizer_with_dataset,
            lowercase=False,
            preprocessor=None,
            token_pattern=None
        )
    tfidf_matrix = vectorizer.fit_transform(raw_texts)


    vectorizer_type = "tfidf"
    file_suffix = f"{dataset_name}_all"
    storage.save_vectorizer(vectorizer, file_suffix, vectorizer_type=vectorizer_type)
    storage.save_tfidf_matrix(tfidf_matrix, file_suffix, vectorizer_type=vectorizer_type)
    storage.save_doc_ids(doc_ids, file_suffix, vectorizer_type=vectorizer_type)


    print(f"[✓] تم بناء وحفظ نموذج TF-IDF لمجموعة البيانات: {dataset_name}")

