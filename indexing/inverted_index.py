import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import mysql.connector
from collections import defaultdict
import joblib

from text_processing.text_preprocessing import get_preprocessed_text_terms

def build_inverted_index(dataset_name: str):
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

    inverted_index = defaultdict(set)

    for doc_id, raw_text in rows:
        if not raw_text or not raw_text.strip():
            print(f"[!] الوثيقة {doc_id} تم تجاوزها (النص الأصلي فارغ).")
            continue

        terms = get_preprocessed_text_terms(raw_text, dataset_name)

        if not terms:
            print(f"[!] الوثيقة {doc_id} تم تجاوزها (لا توجد كلمات بعد المعالجة).")
            continue

        for term in set(terms):
            # هنا لم نعد نحول doc_id إلى نص بل نضيفه كما هو (غالبًا int)
            inverted_index[term].add(doc_id)

    inverted_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}

    output_folder = r"C:\Users\HP\IR-project\indexing\saved_models\inverted_index"
    os.makedirs(output_folder, exist_ok=True)

    index_path = os.path.join(output_folder, f"{dataset_name}_inverted_index.joblib")
    joblib.dump(inverted_index, index_path, compress=3)

    print(f"[✓] تم بناء وحفظ الفهرس المعكوس بصيغة joblib لـ: {dataset_name}")
