import numpy as np
import joblib
from sklearn.preprocessing import normalize
import sys
import os
from scipy.sparse import hstack


project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from storage.vector_storage import load_tfidf_matrix, save_hybrid, load_embeddings

def generate_hybrid(dataset_name: str):
    
    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    embeddings =   load_embeddings(f"{dataset_name}_all")  
    

    tfidf_matrix = normalize(tfidf_matrix)
    embeddings =   normalize(embeddings)

    print("tfidf_matrix.shape:", tfidf_matrix.shape)
    print("embeddings.shape:", embeddings.shape)

    hybrid_vectors = hstack([tfidf_matrix, embeddings])
    print("✅ hybrid_vectors created:", hybrid_vectors.shape)


    #save_hybrid(hybrid_vectors,f"{dataset_name}_all")
    #print('hybrid save')
   

    