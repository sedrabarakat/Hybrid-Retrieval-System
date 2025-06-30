from fastapi import APIRouter, HTTPException
from typing import List
import mysql.connector
from pydantic import BaseModel

router = APIRouter() 

class Document(BaseModel):
    id: int
    text: str

@router.get("/database/build-save-vectorizer", response_model=List[Document])
def build_save_vectorizer_endpoint(dataset_name: str, skip: int = 0, limit: int = 1000):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="ir"
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, processed_text FROM documents WHERE dataset_name = %s LIMIT %s OFFSET %s",
            (dataset_name, limit, skip)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [{"id": row[0], "text": row[1]} for row in rows]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/database/text/", response_model=List[Document])  # Note the trailing slash
def get_documents(dataset_name: str, skip: int = 0, batch_size: int = 1000):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="ir"
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text FROM documents WHERE dataset_name = %s LIMIT %s OFFSET %s",
            (dataset_name, batch_size, skip)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not rows:
            raise HTTPException(status_code=404, detail="No documents found")
            
        return [{"id": row[0], "text": row[1]} for row in rows]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))