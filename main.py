from fastapi import FastAPI
from text_processing.text_processing_controller import router as text_processing_router
from dataBases.database_controller import router as database_router

app = FastAPI()

app.include_router(database_router)
app.include_router(text_processing_router)
