from fastapi import APIRouter, HTTPException
from .text_preprocessing import get_preprocessed_text_terms
import requests
import sys
import os
from typing import List, Dict, Any
from collections import defaultdict
import joblib

router = APIRouter()

@router.get("/text_processing/get_preprocessed_text_terms")
def get_preprocessed_text_terms_request(text: str, dataset_name: str):
    processed_tokens = get_preprocessed_text_terms(text, dataset_name)
    return processed_tokens

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
    
    while True:
        batch = fetch_documents(dataset_name, skip, batch_size)
        if not batch:
            break
        all_docs.extend(batch)
        print(f"Fetched batch {skip//batch_size + 1} ({len(batch)} documents)")
        skip += batch_size
    
    return all_docs

def call_preprocessing_api(text: str, dataset_name: str) -> List[str]:
    """Call the preprocessing API endpoint."""
    try:
        url = f"http://localhost:8000/text_processing/get_preprocessed_text_terms?text={text}&dataset_name={dataset_name}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling preprocessing API: {str(e)}")
        return []

def build_inverted_index(dataset_name: str):
    rows = fetch_all_documents(dataset_name)

    inverted_index = defaultdict(set)

    for doc_id, raw_text in rows:
        if not raw_text or not raw_text.strip():
            print(f"[!] Document {doc_id} skipped (empty raw text).")
            continue

        terms = call_preprocessing_api(raw_text, dataset_name)

        if not terms:
            print(f"[!] Document {doc_id} skipped (no terms after processing).")
            continue

        for term in set(terms):
            inverted_index[term].add(doc_id)

    inverted_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}

    output_folder = r"C:\Users\HP\IR-project\indexing\saved_models\inverted_index"
    os.makedirs(output_folder, exist_ok=True)

    index_path = os.path.join(output_folder, f"{dataset_name}_inverted_index.joblib")
    joblib.dump(inverted_index, index_path, compress=3)

    print(f"[✓] Inverted index built and saved as joblib for: {dataset_name}")