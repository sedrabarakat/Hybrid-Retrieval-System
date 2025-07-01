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

BASE_PATH = "vectorize\saved_models"
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

class LoadDocIdsResponse(BaseModel):
    doc_ids: List[int]

class LoadVectorizerResponse(BaseModel):
    vectorizer_data: str  # base64 encoded serialized vectorizer

@router.get("/load_doc_ids/{name}", response_model=LoadDocIdsResponse)
async def load_doc_ids_api(
    name: str, 
    vectorizer_type: str = "tfidf"
):
    """
    Load document IDs from storage
    
    Args:
        name: Base name of the file (without _doc_ids.joblib suffix)
        vectorizer_type: Subdirectory name (default: "tfidf")
        
    Returns:
        List of document IDs
    """
    try:
        path = os.path.join(_get_dir(vectorizer_type), f"{name}_doc_ids.joblib")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Document IDs file not found at: {path}")
        
        doc_ids = joblib.load(path)
        return {"doc_ids": doc_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/load_vectorizer/{name}", response_model=LoadVectorizerResponse)
async def load_vectorizer_api(
    name: str, 
    vectorizer_type: str = "tfidf"
):
    """
    Load a vectorizer from storage
    
    Args:
        name: Base name of the file (without _vectorizer.joblib suffix)
        vectorizer_type: Subdirectory name (default: "tfidf")
        
    Returns:
        Base64 encoded serialized vectorizer
    """
    try:
        path = os.path.join(_get_dir(vectorizer_type), f"{name}_vectorizer.joblib")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Vectorizer file not found at: {path}")
        
        with open(path, "rb") as f:
            vectorizer_data = base64.b64encode(f.read()).decode('utf-8')
        
        return {"vectorizer_data": vectorizer_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    from fastapi.responses import FileResponse
from fastapi import Query

@router.get("/load_tfidf_matrix/{name}")
async def load_tfidf_matrix_api(
    name: str,
    vectorizer_type: str = Query("tfidf", description="Subdirectory name, e.g., 'tfidf'")
):
    """
    Download the saved TF-IDF matrix (.npz file) from storage.

    Args:
        name: Base name of the file (without _tfidf.npz suffix)
        vectorizer_type: Subdirectory name (default: "tfidf")

    Returns:
        FileResponse to download the .npz file
    """
    try:
        path = os.path.join(_get_dir(vectorizer_type), f"{name}_tfidf.npz")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"TF-IDF matrix file not found at: {path}")

        return FileResponse(path, media_type="application/octet-stream", filename=f"{name}_tfidf.npz")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LoadDocIdsResponse(BaseModel):
    doc_ids: List[int]

@router.get("/load_doc_ids/{name}", response_model=LoadDocIdsResponse)
async def load_doc_ids_api(
    name: str,
    vectorizer_type: str = Query("tfidf", description="Subdirectory name, e.g., 'tfidf'")
):
    """
    Load document IDs from storage.

    Args:
        name: Base name of the file (without '_doc_ids.joblib' suffix)
        vectorizer_type: Subdirectory name (default: 'tfidf')

    Returns:
        List of document IDs
    """
    try:
        path = os.path.join(_get_dir(vectorizer_type), f"{name}_doc_ids.joblib")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Document IDs file not found at: {path}")

        doc_ids = joblib.load(path)
        # Return as list of ints (assuming they are ints)
        return {"doc_ids": doc_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))