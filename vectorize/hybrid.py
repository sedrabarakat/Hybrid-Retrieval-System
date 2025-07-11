import numpy as np
from sklearn.preprocessing import normalize
import sys
import os
from scipy.sparse import hstack
import joblib
from scipy.sparse import csr_matrix


project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from storage.vector_storage import load_tfidf_matrix, load_embeddings

def generate_hybrid(dataset_name: str):
    
    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    embeddings =   load_embeddings(f"{dataset_name}_all")  
    

    tfidf_matrix = normalize(tfidf_matrix)
    embeddings =   normalize(embeddings)

    print("tfidf_matrix.shape:", tfidf_matrix.shape)
    print("embeddings.shape:", embeddings.shape)

    embedding_matrix = csr_matrix(embeddings * 2.0)  
    hybrid_vectors = hstack([tfidf_matrix, embedding_matrix])

    print("✅ hybrid_vectors created:", hybrid_vectors.shape)

   
   

    