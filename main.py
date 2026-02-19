from fastapi import FastAPI, Request
from text_processing.text_processing_controller import router as text_processing_router
from dataBases.database_controller import router as database_router
from storage.vector_storage_controller import router as vector_storage_router
from query_processing.query_processing_controller import router as query_processing_router
from indexing.index_controller import router as indexing_router
from query_processing.hybrid.hybrid_match_and_rank_controller import router as hybrid_match_and_rank_router
from query_processing.embedding.embadding_ranking_controller import router as embadding_match_and_rank_router
from query_processing.tf.ranking_controller import router as tf_match_and_rank_router
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging

app = FastAPI()

app.include_router(database_router)
app.include_router(text_processing_router)
app.include_router(vector_storage_router)
app.include_router(query_processing_router)
app.include_router(indexing_router)
app.include_router(hybrid_match_and_rank_router)
app.include_router(tf_match_and_rank_router)
app.include_router(embadding_match_and_rank_router)

# Mount static files if you have any
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

from query_processing.hybrid.hybrid_match_and_rank_controller import router as hybrid_router
app.include_router(hybrid_router)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
async def search_interface(request: Request):
        # logger.info("Root endpoint accessed")
        # return {"message": "Hello World"}

    return templates.TemplateResponse(
        "hybrid_search.html",
        {"request": request, "base_url_ir": "http://127.0.0.1:8000"}
    )
