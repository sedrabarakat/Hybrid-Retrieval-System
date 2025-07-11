import faiss
import numpy as np
import joblib
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from storage.vector_storage import  load_embeddings    

def build_vector_func(dataset_name:str):
 
 embeddings = load_embeddings(f"{dataset_name}_all")  


 dimension = embeddings.shape[1]

 index = faiss.IndexFlatL2(dimension)

 index.add(embeddings)

 faiss.write_index(index, f"vector_store_index/embedding/{dataset_name}_faiss_index.index")

 print(f"[✓] تم إنشاء وحفظ الفهرس بنجاح. عدد العناصر:", index.ntotal)
