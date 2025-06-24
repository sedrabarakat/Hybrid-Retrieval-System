
from sentence_transformers import SentenceTransformer
from mysql import connector
import joblib
from storage import vector_storage



def generateEmbading(dataset_name:str):
    conn = connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT processed_text FROM documents WHERE dataset_name = %s", (dataset_name,))
    rows = cursor.fetchall()


    documents = [row[0] for row in rows]

    conn.close()

    if not documents:
        print(f"[!] لا يوجد وثائق في مجموعة البيانات: '{dataset_name}'")
        return



    model = SentenceTransformer('all-MiniLM-L6-v2')


    embeddings = model.encode(documents)
    

   
    file_suffix = f"{dataset_name}_all"
    vector_storage.save_embeddings(embeddings, file_suffix)
    

    print(f"[✓] تم بناء وحفظ نموذج TF-IDF لمجموعة البيانات: {dataset_name}")

    
