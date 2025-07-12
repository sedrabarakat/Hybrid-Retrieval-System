from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Optional
from pydantic import BaseModel
from ..tf.ranking import match_and_rank_tfidf
import logging
from ..suggestion_system import QuerySuggester

router = APIRouter(
    prefix="/tf",
    tags=["TF-IDF Ranking"],
    responses={404: {"description": "Not found"}},
)

# Initialize the index loader

class RankingRequest(BaseModel):
    query_text: str
    dataset_name: str
    similarity_threshold: Optional[float] = 0.0001
    top_k: Optional[int] = None
    include_text: Optional[bool] = False

class RankingResponse(BaseModel):
    doc_id: str
    score: float
    text: Optional[str] = None

antique_suggester = QuerySuggester("antique")
beir_suggester = QuerySuggester("beir")

@router.post("/match_and_rank", response_model=Dict[str, RankingResponse])
async def match_and_rank_api(request: RankingRequest):
    suggester = antique_suggester if request.dataset_name == "antique" else beir_suggester
    suggester.log_query(request.query_text)
    """
    Match and rank documents using TF-IDF cosine similarity
    
    Parameters:
    - query_text: The search query
    - dataset_name: Name of the dataset (antique/beir)
    - similarity_threshold: Minimum similarity score (default: 0.0001)
    - top_k: Number of top results to return (optional)
    - include_text: Whether to include document text in response
    
    Returns:
    - Dictionary of document IDs with their scores and optionally text
    """
    try:        
        # Perform the ranking
        ranked_results = match_and_rank_tfidf(
            query=request.query_text,
            dataset_name=request.dataset_name,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k
        )
        
        # Prepare response
        response = {}
        for doc_id, score in ranked_results.items():
            response[doc_id] = RankingResponse(
                doc_id=doc_id,
                score=score,
                text=None  # Will be added below if requested
            )
        
        # Optionally fetch document texts
        if request.include_text:
            from ..tf.ranking import fetch_documents_text_via_api
            doc_ids = list(ranked_results.keys())
            texts = fetch_documents_text_via_api(request.dataset_name, doc_ids)
            for doc_id in response:
                if doc_id in texts:
                    response[doc_id].text = texts[doc_id]
        
        return response
        
    except Exception as e:
        logging.error(f"Error in match_and_rank: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/preload/{dataset_name}")
async def preload_dataset(dataset_name: str):
    """
    Preload dataset index into memory
    
    Parameters:
    - dataset_name: Name of the dataset to preload
    
    Returns:
    - Status message
    """
    try:
        return {"status": "success", "message": f"Dataset {dataset_name} preloaded"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error preloading dataset: {str(e)}"
        )