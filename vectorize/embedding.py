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