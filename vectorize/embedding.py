from sentence_transformers import SentenceTransformer
from mysql import connector
import joblib
import numpy as np



def generateEmbading(dataset_name: str):
    print("[i] بدء الدالة generateEmbading ...", flush=True)

    conn = connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT document_id,process_text2 FROM documents WHERE dataset_name = %s", (dataset_name,))
    rows = cursor.fetchall()

    print(f"[i] عدد الصفوف المسترجعة من قاعدة البيانات: {len(rows)}", flush=True)

    doc_ids = [row[0] for row in rows]
    documents = [row[1] for row in rows]
    conn.close()

    if not documents:
        print(f"[!] لا يوجد وثائق في مجموعة البيانات: '{dataset_name}'")
        return

    model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    embeddings = model.encode(documents, device='cuda', batch_size=64, show_progress_bar=True)

    print(f"[✓] تم بناء وحفظ نموذج embeddings لمجموعة البيانات: {dataset_name}")

    joblib.dump(embeddings, f"{dataset_name}_all_embeddings.joblib")
    joblib.dump(doc_ids, f"{dataset_name}_all_embeddings_doc_ids.joblib")

    np.save(f"{dataset_name}_all_embeddings.npy", embeddings)

    print(f"stores saved")
