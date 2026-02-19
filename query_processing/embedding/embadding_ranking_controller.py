from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from collections import OrderedDict
from ..embedding.embadding_ranking import (
    match_and_rank_embedding,
    match_and_rank_faiss
)
router = APIRouter(
    prefix="/embedding",
    tags=["Embedding Search"],
    responses={404: {"description": "Not found"}},
)

class EmbeddingSearchRequest(BaseModel):
    query_text: str
    dataset_name: str
    similarity_threshold: Optional[float] = 0.3
    top_k: Optional[int] = 10

class FaissSearchRequest(BaseModel):
    query_text: str
    dataset_name: str
    top_k: Optional[int] = 10

class SearchResult(BaseModel):
    doc_id: str
    score: float

@router.post("/search", response_model=Dict[str, SearchResult])
async def embedding_search(request: EmbeddingSearchRequest):
    """
    Perform semantic search using cosine similarity on embeddings
    
    Parameters:
    - query_text: Search query text
    - dataset_name: Name of dataset (antique/beir)
    - similarity_threshold: Minimum similarity score (default: 0.3)
    - top_k: Number of top results to return (default: 10)
    
    Returns:
    - Dictionary of document IDs with their similarity scores
    """
    try:
        results = match_and_rank_embedding(
            query_text=request.query_text,
            dataset_name=request.dataset_name,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k
        )
        return {doc_id: SearchResult(doc_id=doc_id, score=score) 
                for doc_id, score in results.items()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding search failed: {str(e)}"
        )

@router.post("/faiss/search", response_model=Dict[str, SearchResult])
async def faiss_search(request: FaissSearchRequest):
    """
    Perform approximate nearest neighbor search using FAISS index
    
    Parameters:
    - query_text: Search query text
    - dataset_name: Name of dataset (antique/beir)
    - top_k: Number of top results to return (default: 10)
    
    Returns:
    - Dictionary of document IDs with their similarity scores
    """
    try:
        results = match_and_rank_faiss(
            query_text=request.query_text,
            dataset_name=request.dataset_name,
            top_k=request.top_k
        )
        return {doc_id: SearchResult(doc_id=doc_id, score=score) 
                for doc_id, score in results.items()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"FAISS search failed: {str(e)}"
        )

@router.get("/datasets")
async def available_datasets():
    """
    List available datasets for embedding search
    """
    return {
        "datasets": ["antique", "beir"],
        "default": "antique"
    }from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from collections import OrderedDict
from ..suggestion_system import QuerySuggester
from ..embedding.embadding_ranking import (
    match_and_rank_embedding,
    match_and_rank_faiss
)
router = APIRouter(
    prefix="/embedding",
    tags=["Embedding Search"],
    responses={404: {"description": "Not found"}},
)
antique_suggester = QuerySuggester("antique")
beir_suggester = QuerySuggester("beir")
class EmbeddingSearchRequest(BaseModel):
    query_text: str
    dataset_name: str
    similarity_threshold: Optional[float] = 0.3
    top_k: Optional[int] = 10

class FaissSearchRequest(BaseModel):
    query_text: str
    dataset_name: str
    top_k: Optional[int] = 10

class SearchResult(BaseModel):
    doc_id: str
    score: float

@router.post("/search", response_model=Dict[str, SearchResult])
async def embedding_search(request: EmbeddingSearchRequest):
    suggester = antique_suggester if request.dataset_name == "antique" else beir_suggester
    suggester.log_query(request.query_text)
    """
    Perform semantic search using cosine similarity on embeddings
    
    Parameters:
    - query_text: Search query text
    - dataset_name: Name of dataset (antique/beir)
    - similarity_threshold: Minimum similarity score (default: 0.3)
    - top_k: Number of top results to return (default: 10)
    
    Returns:
    - Dictionary of document IDs with their similarity scores
    """
    try:
        results = match_and_rank_embedding(
            query_text=request.query_text,
            dataset_name=request.dataset_name,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k
        )
        return {doc_id: SearchResult(doc_id=doc_id, score=score) 
                for doc_id, score in results.items()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding search failed: {str(e)}"
        )

@router.post("/faiss/search", response_model=Dict[str, SearchResult])
async def faiss_search(request: FaissSearchRequest):
    suggester = antique_suggester if request.dataset_name == "antique" else beir_suggester
    suggester.log_query(request.query_text)
    """
    Perform approximate nearest neighbor search using FAISS index
    
    Parameters:
    - query_text: Search query text
    - dataset_name: Name of dataset (antique/beir)
    - top_k: Number of top results to return (default: 10)
    
    Returns:
    - Dictionary of document IDs with their similarity scores
    """
    try:
        results = match_and_rank_faiss(
            query_text=request.query_text,
            dataset_name=request.dataset_name,
            top_k=request.top_k
        )
        return {doc_id: SearchResult(doc_id=doc_id, score=score) 
                for doc_id, score in results.items()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"FAISS search failed: {str(e)}"
        )

@router.get("/datasets")
async def available_datasets():
    """
    List available datasets for embedding search
    """
    return {
        "datasets": ["antique", "beir"],
        "default": "antique"
    }