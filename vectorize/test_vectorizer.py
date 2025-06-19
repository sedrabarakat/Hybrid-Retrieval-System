import sys
import os

# إضافة جذر المشروع (parent directory) للمسار
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing.text_preprocessing import get_preprocessed_text_terms
import storage.vector_storage as storage  # تأكد إن المسار صحيح

# دالة تهيئة توكنيزر خاصة بالداتاست
def make_tokenizer(dataset_name):
    def tokenizer(text):
        return get_preprocessed_text_terms(text, dataset_name)
    return tokenizer

def build_save_vectorizer_first_doc_only(dataset_name: str):
    # الاتصال بقاعدة البيانات
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM documents WHERE dataset_name = %s LIMIT 1", (dataset_name,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        print(f"No document found for dataset '{dataset_name}'")
        return

    raw_texts = [row[0]]  # نص أول مستند فقط

    # نهيئ التوكنيزر
    tokenizer = make_tokenizer(dataset_name)

    # تعريف vectorizer مع التوكنيزر الخارجي (مش لامبدا)
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        lowercase=False,
        preprocessor=None,
        token_pattern=None
    )

    # تدريب الـ TF-IDF على النصوص
    tfidf_matrix = vectorizer.fit_transform(raw_texts)

    # حفظ النموذج والمصفوفة
    storage.save_vectorizer(vectorizer, dataset_name + "_test")
    storage.save_tfidf_matrix(tfidf_matrix, dataset_name + "_test")

    print(f"[✓] Vectorizer and matrix for '{dataset_name}' (first doc only) saved successfully.")

# تشغيل على عدة داتاسات
datasets = ["beir", "antique", "quora"]

for ds in datasets:
    build_save_vectorizer_first_doc_only(ds)
