import sys
import os
import mysql.connector

# اضيف جذر المشروع إلى sys.path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corpus import get_corpus
from app.text_processing.text_preprocessing import get_preprocessed_text_terms

# إعداد الاتصال بقاعدة البيانات
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  
    database="ir" 
)
cursor = conn.cursor()

datasets = ["antique", "beir"]

for dataset_name in datasets:
    corpus = get_corpus(dataset_name)
    first_doc = next(iter(corpus.items()))
    doc_id, text = first_doc

    processed = get_preprocessed_text_terms(text, dataset_name)
    processed_str = " ".join(processed)

    # ✨ صار داخل اللوب
    cursor.execute(
        "INSERT INTO documents (document_id, dataset_name, processed_text, text) VALUES (%s, %s, %s, %s)",
        (doc_id, dataset_name, processed_str, text)
    )


conn.commit()
cursor.close()
conn.close()
