import sys
import types

sys.modules['TF_IDF'] = types.ModuleType('TF_IDF')
import joblib
import os
from text_processing.text_preprocessing import get_preprocessed_text_terms

class QueryProcessor:
    def __init__(self, dataset_name: str):
        if dataset_name not in ["antique", "beir"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_name = dataset_name
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # مستوى IR-project
        model_dir = os.path.join(base_dir, "vectorize", "saved_models", "tfidf")

        self.vectorizer = joblib.load(os.path.join(model_dir, f"{dataset_name}_all_vectorizer.joblib"))
        self.doc_ids = joblib.load(os.path.join(model_dir, f"{dataset_name}_all_doc_ids.joblib"))
    
    def process(self, query_text: str):
        tokens = get_preprocessed_text_terms(query_text, self.dataset_name)
        processed_query_text = " ".join(tokens)
        query_vec = self.vectorizer.transform([processed_query_text])
        return query_vec

def query_service(dataset_name, query_text):
    processor = QueryProcessor(dataset_name)
    return processor.process(query_text)
