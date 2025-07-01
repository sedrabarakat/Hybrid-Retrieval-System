from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import mysql.connector
from pydantic import BaseModel

router = APIRouter()

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "ir"
}

# Pydantic Models
class Document(BaseModel):
    id: int
    text: str

class QueryModel(BaseModel):
    id: int
    text: str

class DocumentIDMapping(BaseModel):
    internal_id: int
    original_document_id: str

# Database Connection Helper
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# API Endpoints
@router.get("/database/processed_text", response_model=List[Document])
def build_save_vectorizer_endpoint(
    dataset_name: str, 
    skip: int = 0, 
    limit: int = 1000
):
    """Retrieve processed documents for vectorizer building"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, processed_text FROM documents WHERE dataset_name = %s LIMIT %s OFFSET %s",
            (dataset_name, limit, skip)
        )
        rows = cursor.fetchall()
        return [{"id": row[0], "text": row[1]} for row in rows]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@router.get("/database/text/", response_model=List[Document])
def get_documents(
    dataset_name: str, 
    skip: int = 0, 
    batch_size: int = 1000
):
    """Retrieve raw document texts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text FROM documents WHERE dataset_name = %s LIMIT %s OFFSET %s",
            (dataset_name, batch_size, skip)
        )
        rows = cursor.fetchall()
        
        if not rows:
            raise HTTPException(status_code=404, detail="No documents found")
            
        return [{"id": row[0], "text": row[1]} for row in rows]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@router.get("/database/queries/", response_model=Dict[str, str])
def get_queries(
    dataset_name: str, 
    limit: int = 3
):
    """Retrieve search queries"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, query_text FROM queries WHERE dataset_name = %s LIMIT %s",
            (dataset_name, limit)
        )
        rows = cursor.fetchall()
        
        if not rows:
            raise HTTPException(status_code=404, detail="No queries found")
            
        return {str(row[0]): row[1] for row in rows}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@router.get("/database/document-id-mapping/", response_model=Dict[int, str])
def get_document_id_mapping(dataset_name: str):
    """Retrieve mapping between internal and original document IDs"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, document_id FROM documents WHERE dataset_name = %s",
            (dataset_name,)
        )
        rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No documents found")

        return {int(row['id']): str(row['document_id']) for row in rows}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@router.get("/database/documents-text/", response_model=Dict[int, str])
def get_documents_text(
    dataset_name: str,
    doc_ids: Optional[str] = Query(None, description="Comma separated list of document IDs")
):
    """Retrieve document texts by their IDs"""
    try:
        if not doc_ids:
            return {}

        # Parse comma-separated string into list of ints
        try:
            doc_ids_list = [int(id_str) for id_str in doc_ids.split(",") if id_str.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="doc_ids must be a comma-separated list of integers")

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        placeholders = ','.join(['%s'] * len(doc_ids_list))
        query = f"""
            SELECT id, text
            FROM documents
            WHERE id IN ({placeholders}) AND dataset_name = %s
        """
        params = (*doc_ids_list, dataset_name)
        cursor.execute(query, params)
        rows = cursor.fetchall()

        return {row['id']: row['text'] for row in rows}

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()