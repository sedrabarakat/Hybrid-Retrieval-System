from fastapi import FastAPI
from text_processing.text_processing_controller import router as text_processing_router
from dataBases.database_controller import router as database_router
from storage.vector_storage_controller import router as vector_storage_router
from query_processing.query_processing_controller import router as query_processing_router
from indexing.index_controller import router as indexing_router

app = FastAPI()

app.include_router(database_router)
app.include_router(text_processing_router)
app.include_router(vector_storage_router)
app.include_router(query_processing_router)
app.include_router(indexing_router)
