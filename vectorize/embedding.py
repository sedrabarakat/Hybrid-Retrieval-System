<<<<<<< HEAD
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
=======
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def fetch_documents(dataset_name: str, skip: int, limit: int) -> List[Dict[str, Any]]:
    """Fetch a batch of documents from the API."""
    url = f"http://localhost:8000/database/processed_text?dataset_name={dataset_name}&skip={skip}&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()  # Raises exception for 4XX/5XX status codes
    return response.json()

def save_embeddings(dataset_name: str, embeddings: List[List[float]]) -> None:
    """Save embeddings to the storage API."""
    response = requests.post(
        "http://localhost:8000/storage/save_embeddings/",
        json={
            "name": f"{dataset_name}_all",
            "embeddings": embeddings
        }
    )
    response.raise_for_status()

def generate_embedding(dataset_name: str, batch_size: int = 1000) -> None:
    """Generate and save embeddings for all documents in a dataset."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_texts = []
    skip = 0

    print(f"Starting embedding generation for dataset: {dataset_name}")
    
    # Fetch all documents in batches
    loop = 0
    while True:
        loop = loop + 1
        print(f"Fetching batch starting at document {skip}")
        
        try:
            documents = fetch_documents(dataset_name, skip, batch_size)
            if not documents:
                break
            if loop == 2:
                break
                
            all_texts.extend(doc["text"] for doc in documents)
            print(f"Processed batch {skip//batch_size + 1} - {len(documents)} documents")
            skip += batch_size
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching documents: {str(e)}")
            return

    if not all_texts:
        print("No documents found to process!")
        return

    print(f"Total documents loaded: {len(all_texts)}")
    
    # Generate embeddings
    print("Generating embeddings...")
    try:
        embeddings = model.encode(all_texts[:10]).tolist()
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return

    # Save embeddings
    print("Saving embeddings...")
    try:
        save_embeddings(dataset_name, embeddings)
        print(f"Successfully saved embeddings for dataset: {dataset_name}_all")
    except requests.exceptions.RequestException as e:
        print(f"Error saving embeddings: {str(e)}")

if __name__ == "__main__":
    generate_embedding("your_dataset_name")
>>>>>>> apis
