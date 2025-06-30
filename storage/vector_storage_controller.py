from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os
from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz
import pickle
import base64
import joblib
from fastapi.responses import FileResponse

BASE_PATH = "saved_models"
router = APIRouter(prefix="/storage")

def _get_dir(vectorizer_type: str) -> str:
    dir_path = os.path.join(BASE_PATH, vectorizer_type)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

class EmbeddingsRequest(BaseModel):
    name: str
    embeddings: List[List[float]]
    vectorizer_type: str = "embedding"

class VectorizerRequest(BaseModel):
    name: str
    data: str  # base64 encoded
    vectorizer_type: str = "tfidf"

class MatrixRequest(BaseModel):
    name: str
    data: str  # base64 encoded
    vectorizer_type: str = "tfidf"

class HybridRequest(BaseModel):
    name: str
    data: str  # base64 encoded
    vectorizer_type: str = "Hybrid"

@router.post("/save_embeddings/")
async def save_embeddings(request: EmbeddingsRequest):
    try:
        path = os.path.join(_get_dir(request.vectorizer_type), f"{request.name}_embeddings.npy")
        np.save(path, np.array(request.embeddings))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save_vectorizer/")
async def save_vectorizer(request: VectorizerRequest):
    try:
        data = base64.b64decode(request.data.encode('utf-8'))
        vectorizer = pickle.loads(data)
        
        path = os.path.join(_get_dir(request.vectorizer_type), f"{request.name}_vectorizer.joblib")
        joblib.dump(vectorizer, path, compress=3)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save_tfidf_matrix/")
async def save_tfidf_matrix(request: MatrixRequest):
    try:
        data = base64.b64decode(request.data.encode('utf-8'))
        matrix = pickle.loads(data)
        
        path = os.path.join(_get_dir(request.vectorizer_type), f"{request.name}_tfidf.npz")
        save_npz(path, matrix)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save_hybrid/")
async def save_hybrid(request: HybridRequest):
    try:
        data = base64.b64decode(request.data.encode('utf-8'))
        hybrid = pickle.loads(data)
        
        path = os.path.join(_get_dir(request.vectorizer_type), f"{request.name}_hybrid.joblib")
        joblib.dump(hybrid, path, compress=3)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_tfidf/{name}")
async def download_tfidf_api(name: str):
    base_name = name.replace('_tfidf', '')
    path = os.path.join("saved_models", "tfidf", f"{base_name}_tfidf.npz")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{base_name}_tfidf.npz")
    
@router.get("/load_embeddings/{name}")
async def load_embeddings_api(name: str, vectorizer_type: str = "embedding"):
    try:
        path = os.path.join(_get_dir(vectorizer_type), f"{name}_embeddings.npy")
        embeddings = np.load(path)
        return {"data": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/load_hybrid/{name}")
async def load_hybrid_api(name: str, vectorizer_type: str = "Hybrid"):
    try:
        path = os.path.join(_get_dir(vectorizer_type), f"{name}_hybrid.joblib")
        hybrid = joblib.load(path)
        
        if scipy.sparse.issparse(hybrid):
            hybrid_coo = hybrid.tocoo()
            return {
                "data": hybrid_coo.data.tolist(),
                "row": hybrid_coo.row.tolist(),
                "col": hybrid_coo.col.tolist(),
                "shape": hybrid_coo.shape,
                "is_sparse": True
            }
        else:
            return {
                "data": hybrid.tolist(),
                "is_sparse": False
            }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))