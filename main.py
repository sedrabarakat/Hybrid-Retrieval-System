from fastapi import FastAPI, Request
from dataBases.database_controller import router as database_router
from query_processing.hybrid.hybrid_match_and_rank_controller import router as hybrid_match_and_rank_router
from query_processing.embedding.embadding_ranking_controller import router as embadding_match_and_rank_router
from query_processing.tf.ranking_controller import router as tf_match_and_rank_router
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(database_router)
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
