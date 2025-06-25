import os
import joblib

def load_inverted_index(dataset_name: str):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(project_root, "indexing", "saved_models", "inverted_index", f"{dataset_name}_inverted_index.joblib")

    if not os.path.exists(path):
        raise FileNotFoundError(f"لم يتم العثور على الفهرس المعكوس: {path}")
    
    return joblib.load(path)
