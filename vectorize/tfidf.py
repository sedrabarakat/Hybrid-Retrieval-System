import sys
import os
from functools import partial
import numpy as np
from scipy import sparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing.text_preprocessing import clean_and_tokenize_text
from storage.vector_storage import save_tfidf_matrix,save_vectorizer,save_tfidf_doc_ids


def tokenizer(text, dataset_name): return clean_and_tokenize_text(text, dataset_name)


def build_save_vectorizer(dataset_name: str):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT document_id, text FROM documents WHERE dataset_name = %s ", (dataset_name,))
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
            token_pattern=None,
            norm='l2'
        )
    
    tfidf_matrix = vectorizer.fit_transform(raw_texts)

  
    print(f"[✓] تم بناء وحفظ نموذج embeddings لمجموعة البيانات: {dataset_name}")

    save_vectorizer(dataset_name=dataset_name,model=vectorizer)
    save_tfidf_matrix(name=f"{dataset_name}_all",matrix=tfidf_matrix)
    save_tfidf_doc_ids(dataset_name=dataset_name,doc_ids=doc_ids)


    print(f"[✓] تم بناء وحفظ نموذج TF-IDF لمجموعة البيانات: {dataset_name}")

