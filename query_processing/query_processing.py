

import sys
import vectorize.tokenizer_definition 
import requests
import base64
import pickle
sys.modules["TF_IDF"] = vectorize.tokenizer_definition

class QueryProcessor:

    def api_get_preprocessed_text_terms(text: str, dataset_name: str):
        """
        Call the text preprocessing API to get tokens.
        """
        endpoint = f"http://localhost:8000/text_processing/get_preprocessed_text_terms"
        params = {"text": text, "dataset_name": dataset_name}
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def api_load_vectorizer(name: str, vectorizer_type="tfidf"):
        """
        Load vectorizer from API, decode base64 and unpickle.
        """
        endpoint = f"http://localhost:8000/storage/load_vectorizer/{name}"
        params = {"vectorizer_type": vectorizer_type}
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        vectorizer_data_b64 = data.get("vectorizer_data")
        if not vectorizer_data_b64:
            raise ValueError("No vectorizer data returned from API")
        vectorizer_bytes = base64.b64decode(vectorizer_data_b64.encode("utf-8"))
        vectorizer = pickle.loads(vectorizer_bytes)
        return vectorizer

    def api_load_doc_ids(name: str, vectorizer_type="tfidf"):
        """
        Load doc_ids from API.
        """
        endpoint = f"http://localhost:8000/storage/load_doc_ids/{name}"
        params = {"vectorizer_type": vectorizer_type}
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("doc_ids", [])

    def __init__(self, dataset_name: str):
        if dataset_name not in ["antique", "beir"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name
        file_prefix = f"{dataset_name}_all"

        # تحميل vectorizer
        self.vectorizer = self.api_load_vectorizer(file_prefix)

        # تعديل دالة tokenizer داخل vectorizer لتدعو دالة المعالجة المسبقة مع dataset_name
        def tokenizer_no_arg(text):
            return self.api_get_preprocessed_text_terms(text, self.dataset_name)

        self.vectorizer.tokenizer = tokenizer_no_arg

        # تحميل معرّفات المستندات
        self.doc_ids = self.api_load_doc_ids(file_prefix)

    def process(self, query_text: str):
        tokens = self.api_get_preprocessed_text_terms(query_text, self.dataset_name)
        processed_query_text = " ".join(tokens)
        query_vec = self.vectorizer.transform([processed_query_text])
        return query_vec, tokens
    

    
def process(dataset_name: str, query_text: str):
    qp = QueryProcessor(dataset_name)
    return qp.process(query_text)
