from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from query_processing.query_processing import process


router = APIRouter()

# Define the request body schema
class QueryRequest(BaseModel):
    dataset_name: str
    query_text: str

@router.post("/process_query/")
def process_query(request: QueryRequest):
    try:
        query_vec, tokens = process(request.dataset_name, request.query_text)
        # Convert query_vec to a list or serializable format if needed
        return {
            "tokens": tokens,
            "query_vector": query_vec.toarray().tolist()  # Assuming it's a sparse matrix
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
