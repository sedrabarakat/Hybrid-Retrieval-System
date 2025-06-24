
import sys
import os

# أضف جذر المشروع لمسار بايثون
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# بعد إضافة الجذر، يمكن استيراد مجلد storage بشكل صحيح
import storage.vector_storage as storage
from sentence_transformers import SentenceTransformer
from mysql import connector
import joblib
import storage.vector_storage as storage


def generateEmbading(dataset_name:str):
    print("[i] بدء الدالة generateEmbading ...", flush=True)

    conn = connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT processed_text FROM documents WHERE dataset_name = %s", (dataset_name,))
    rows = cursor.fetchall()

    print(f"[i] عدد الصفوف المسترجعة من قاعدة البيانات: {len(rows)}", flush=True)

    documents = [row[0] for row in rows]

    conn.close()

    if not documents:
        print(f"[!] لا يوجد وثائق في مجموعة البيانات: '{dataset_name}'")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)

    file_suffix = f"{dataset_name}_all"
    storage.save_embeddings(embeddings, file_suffix)

    print(f"[✓] تم بناء وحفظ نموذج embeddings لمجموعة البيانات: {dataset_name}")