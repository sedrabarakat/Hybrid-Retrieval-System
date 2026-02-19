<<<<<<< HEAD
import sys
import os
from functools import partial
import numpy as np
from scipy import sparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing.text_preprocessing import get_preprocessed_text_terms
from storage.vector_storage import save_tfidf_matrix,save_vectorizer,save_tfidf_doc_ids


def tokenizer(text, dataset_name): return get_preprocessed_text_terms(text, dataset_name)


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
=======
import nltk
import os
import sys
from functools import partial
from typing import List, Dict, Any, Optional, Tuple
import requests
from requests.exceptions import RequestException
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle
import base64
import json
import shutil

# ====================== NLTK SETUP WITH CORRUPTION HANDLING ======================
# nltk_data_path = "C:/Users/Eng Islam/AppData/Roaming/nltk_data"

# def clean_and_setup_nltk():
#     """Clean corrupted data and setup NLTK properly"""
#     # Clean up if exists
#     if os.path.exists(nltk_data_path):
#         try:
#             shutil.rmtree(nltk_data_path)
#             print("Cleaned up corrupted NLTK data")
#         except Exception as e:
#             print(f"Error cleaning NLTK data: {str(e)}")
    
#     # Create fresh directory
#     os.makedirs(nltk_data_path, exist_ok=True)
#     nltk.data.path.append(nltk_data_path)
    
#     # Define required packages
#     REQUIRED_PACKAGES = {
#         'stopwords': 'corpora/stopwords',
#         'punkt': 'tokenizers/punkt',
#         'wordnet': 'corpora/wordnet',
#         'omw-1.4': 'corpora/omw-1.4',
#         'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
#     }
    
#     # Download packages with verification
#     for pkg_name, pkg_path in REQUIRED_PACKAGES.items():
#         max_attempts = 2
#         for attempt in range(max_attempts):
#             try:
#                 # Try to find existing package
#                 nltk.data.find(pkg_path)
#                 print(f"Found existing {pkg_name}")
#                 break
#             except LookupError:
#                 try:
#                     print(f"Downloading {pkg_name} (attempt {attempt + 1})...")
#                     nltk.download(pkg_name, download_dir=nltk_data_path)
                    
#                     # Verify download
#                     nltk.data.find(pkg_path)
#                     print(f"Successfully downloaded and verified {pkg_name}")
#                     break
#                 except Exception as e:
#                     if attempt == max_attempts - 1:
#                         print(f"Failed to download {pkg_name}: {str(e)}")
#                         print(f"Please manually download from https://www.nltk.org/nltk_data/")
#                     continue

# # Initialize NLTK with cleanup
# clean_and_setup_nltk()

# ====================== PROJECT IMPORTS ======================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from text_processing.text_preprocessing import get_preprocessed_text_terms

# ====================== DOCUMENT PROCESSING ======================
def fetch_documents(dataset_name: str, skip: int, limit: int) -> List[Dict[str, Any]]:
    """Fetch a batch of documents from the API."""
    try:
        url = f"http://localhost:8000/database/text?dataset_name={dataset_name}&skip={skip}&limit={limit}"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching documents: {str(e)}")
        return []

def fetch_all_documents(dataset_name: str, batch_size: int = 1000) -> List[Dict[str, Any]]:
    """Fetch all documents in batches."""
    all_docs = []
    skip = 0
    
    loop = 0
    while True:
        loop = loop +1
        batch = fetch_documents(dataset_name, skip, batch_size)
        if not batch:
            break
        if loop == 2:
            break
        all_docs.extend(batch)
        print(f"Fetched batch {skip//batch_size + 1} ({len(batch)} documents)")
        skip += batch_size
    
    return all_docs

# ====================== VECTORIZER FUNCTIONS ======================
def serialize_object(obj: object) -> str:
    """Serialize object to base64 string."""
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def save_vectorizer_artifacts(
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    dataset_name: str
) -> bool:
    """Save vectorizer and matrix to storage API with proper error handling."""
    headers = {'Content-Type': 'application/json'}
    
    try:
        # Save vectorizer
        vectorizer_payload = {
            "name": f"{dataset_name}_vectorizer",
            "data": serialize_object(vectorizer),
            "vectorizer_type": "tfidf"  # Add this required field
        }
        response = requests.post(
            "http://localhost:8000/storage/save_vectorizer/",
            json=vectorizer_payload,  # Use json parameter instead of data
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        
        # Save matrix
        matrix_payload = {
            "name": f"{dataset_name}_matrix",
            "data": serialize_object(matrix),
            "vectorizer_type": "tfidf"  # Add this required field
        }
        response = requests.post(
            "http://localhost:8000/storage/save_tfidf_matrix/",
            json=matrix_payload,  # Use json parameter instead of data
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        
        return True
        
    except RequestException as e:
        print(f"Error saving artifacts: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server response: {e.response.text}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False
    
        
def build_tfidf_model(dataset_name: str) -> Optional[Tuple[TfidfVectorizer, csr_matrix]]:
    """Build and return TF-IDF model."""
    print(f"Building TF-IDF model for {dataset_name}")
    
    documents = fetch_all_documents(dataset_name)
    if not documents:
        print("No documents found")
        return None
    
    texts = [doc["text"] for doc in documents[:10] if doc.get("text", "").strip()]
    if not texts:
        print("No valid texts found")
        return None
    
    print(f"Processing {len(texts)} documents")
    
    try:
        tokenizer = partial(get_preprocessed_text_terms, dataset_name=dataset_name)
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
>>>>>>> apis
            lowercase=False,
            preprocessor=None,
            token_pattern=None,
            norm='l2'
        )
<<<<<<< HEAD
    
    tfidf_matrix = vectorizer.fit_transform(raw_texts)

  
    print(f"[✓] تم بناء وحفظ نموذج embeddings لمجموعة البيانات: {dataset_name}")

    save_vectorizer(dataset_name=dataset_name,model=vectorizer)
    save_tfidf_matrix(name=f"{dataset_name}_all",matrix=tfidf_matrix)
    save_tfidf_doc_ids(dataset_name=dataset_name,doc_ids=doc_ids)


    print(f"[✓] تم بناء وحفظ نموذج TF-IDF لمجموعة البيانات: {dataset_name}")

=======
        matrix = vectorizer.fit_transform(texts)
        return vectorizer, matrix
    except Exception as e:
        print(f"Model building failed: {str(e)}")
        return None

# ====================== MAIN EXECUTION ======================
def build_save_vectorizer(dataset_name: str):
    """Orchestrate the vectorization pipeline."""
    if model := build_tfidf_model(dataset_name):
        vectorizer, matrix = model
        if save_vectorizer_artifacts(vectorizer, matrix, dataset_name):
            print(f"Successfully saved TF-IDF model for {dataset_name}")
        else:
            print(f"Failed to save model for {dataset_name}")

if __name__ == "__main__":
    build_save_vectorizer("your_dataset_name")
>>>>>>> apis
