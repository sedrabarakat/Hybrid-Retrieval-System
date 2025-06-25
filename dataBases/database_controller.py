from fastapi import APIRouter, HTTPException
from typing import Dict
from typing import List
import mysql.connector
from pydantic import BaseModel

router = APIRouter()

def build_save_vectorizer(dataset_name: str) -> Dict:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="ir"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM documents WHERE dataset_name = %s", (dataset_name,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return rows


class Document(BaseModel):
    id: int
    text: str

@router.get("/database/build-save-vectorizer", response_model=List[Document])
def build_save_vectorizer_endpoint(dataset_name: str):
    try:
        rows = build_save_vectorizer(dataset_name)
        result = [{"id": row[0], "text": row[1]} for row in rows]
        return result
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))