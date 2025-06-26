import mysql.connector
import argparse
from storage.vector_storage import save_doc_ids
import os
import joblib
import numpy as np
import scipy.sparse

def fetch_and_save_doc_ids(dataset_name: str):
    # الاتصال بقاعدة البيانات
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

    doc_ids = []
    skipped = 0

    for doc_id, text in rows:
        if not text or not text.strip():
            skipped += 1
            continue
        doc_ids.append(doc_id)

    print(f"[✓] عدد المستندات المقبولة: {len(doc_ids)}")
    if skipped > 0:
        print(f"[!] عدد المستندات المُتجاهلة (نص فارغ): {skipped}")

 

    # Use the absolute path to avoid any confusion
    save_dir = r"C:\Users\HP\IR-project\vectorize\saved_models\tfidf"
    # Make sure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Example: Save a joblib model
    model = ...  # النموذج الذي تريد حفظه
    joblib.dump(model, os.path.join(save_dir, "my_model.joblib"))

    # Example: Save a sparse matrix
    data = ...  # البيانات التي تريد حفظها
    scipy.sparse.save_npz(os.path.join(save_dir, "my_data.npz"), data)

    # Example: Save a numpy array
    np.save(os.path.join(save_dir, "my_array.npy"), data.toarray())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save doc_ids for a dataset.")
    parser.add_argument("--dataset", required=True, help="اسم مجموعة البيانات (dataset_name)")
    args = parser.parse_args()
    fetch_and_save_doc_ids(args.dataset)
