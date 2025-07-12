import sys
import os
import mysql.connector

# أضف جذر المشروع إلى sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from corpus import get_corpus
from text_processing.text_preprocessing import get_preprocessed_text_terms

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="ir"
)
cursor = conn.cursor()

datasets = ["antique", "beir", "qora"]

for dataset_name in datasets:
    print(f"جاري معالجة وتخزين بيانات '{dataset_name}'...")
    corpus = get_corpus(dataset_name)
    

    for doc_id, text in corpus.items():
        processed = get_preprocessed_text_terms(text, dataset_name)
        processed_str = " ".join(processed)

        
        cursor.execute(
            "INSERT INTO documents (document_id, dataset_name, processed_text, text) VALUES (%s, %s, %s, %s)",
            (doc_id, dataset_name, processed_str, text)
        )
      
    
    conn.commit()
   

cursor.close()
conn.close()
