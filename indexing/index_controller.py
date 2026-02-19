from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from indexing.inverted_index_loader import load_inverted_index
from pydantic import BaseModel

class InvertedIndexRequest(BaseModel):
    dataset_name: str


router = APIRouter()

@router.get("/load_inverted_index/")
def api_load_inverted_index(dataset_name: str):
    try:
        inverted_index = load_inverted_index(dataset_name)
        return {"inverted_index": inverted_index}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
