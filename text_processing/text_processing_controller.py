from fastapi import APIRouter, HTTPException
from .text_preprocessing import get_preprocessed_text_terms


router = APIRouter()

@router.get("/text_processing/get_preprocessed_text_terms")
def get_preprocessed_text_terms_request(text: str, dataset_name: str):
    processed_tokens = get_preprocessed_text_terms(text, dataset_name)
    return processed_tokens
