from fastapi import APIRouter, HTTPException
from typing import Dict, List
from query_processing.hybrid.hybrid_match_and_rank import match_and_rank_hybrid
from pydantic import BaseModel
from ..suggestion_system import QuerySuggester

router = APIRouter(
    prefix="/hybrid",
    tags=["Hybrid Search"],
    responses={404: {"description": "Not found"}},
)
# Initialize suggesters for each dataset
antique_suggester = QuerySuggester("antique")
beir_suggester = QuerySuggester("beir")

class SearchRequest(BaseModel):
    query_text: str

class SuggestionRequest(BaseModel):
    partial_query: str
    dataset_name: str

@router.post("/search/{dataset_name}", response_model=Dict[str, float])
async def hybrid_search(
    dataset_name: str,
    request: SearchRequest,
    top_k: int = 10,
    similarity_threshold: float = 0.3
):
    """
    Perform hybrid search on a specified dataset.

    Parameters:
    - dataset_name: Name of the dataset to search
    - request: Contains query_text in request body
    - top_k: Number of top results to return (default: 10)
    - similarity_threshold: Minimum similarity score for results (default: 0.3)

    Returns:
    - Ordered dictionary of document IDs and their similarity scores

    """
    suggester = antique_suggester if dataset_name == "antique" else beir_suggester
    suggester.log_query(request.query_text)
    try:

        results = match_and_rank_hybrid(
            query_text=request.query_text,
            dataset_name=dataset_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing hybrid search: {str(e)}"
        )
    
    
@router.get("/datasets/{dataset_name}/healthcheck")
async def healthcheck(dataset_name: str):
    """
    Health check endpoint to verify the dataset is available for hybrid search.
    """
    try:
        # Try to load a small sample to verify the dataset is accessible
        _ = match_and_rank_hybrid(
            query_text="test",
            dataset_name=dataset_name,
            top_k=1
        )
        return {"status": "healthy", "dataset": dataset_name}
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_name} not available for hybrid search: {str(e)}"
        )
    
@router.post("/suggest", response_model=List[str])
async def get_suggestions(request: SuggestionRequest):
    """New endpoint for suggestions"""
    suggester = antique_suggester if request.dataset_name == "antique" else beir_suggester
    return suggester.get_suggestions(request.partial_query)