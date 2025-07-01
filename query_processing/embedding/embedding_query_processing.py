# embedding_query_processing.py

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from sentence_transformers import SentenceTransformer
from storage.vector_storage import load_doc_ids, load_embeddings_joblib
from text_processing.text_preprocessing import get_preprocessed_text_terms


class EmbeddingQueryProcessor:
    _cache_models = {}
    _cache_doc_ids = {}
    _cache_embeddings = {}

    def __init__(self, dataset_name: str, model_name: str = "all-MiniLM-L6-v2"):
        if dataset_name not in ["antique", "beir"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.model_name = model_name
        file_prefix = f"{dataset_name}_all"

        # تحميل الموديل مرة واحدة
        if model_name in EmbeddingQueryProcessor._cache_models:
            print(f"[CACHE] Using cached embedding model {model_name}")
            self.model = EmbeddingQueryProcessor._cache_models[model_name]
        else:
            print(f"[DISK] Loading embedding model {model_name}")
            model = SentenceTransformer(model_name)
            EmbeddingQueryProcessor._cache_models[model_name] = model
            self.model = model

        # تحميل doc_ids
        if dataset_name in EmbeddingQueryProcessor._cache_doc_ids:
            self.doc_ids = EmbeddingQueryProcessor._cache_doc_ids[dataset_name]
        else:
            doc_ids = load_doc_ids(file_prefix)
            EmbeddingQueryProcessor._cache_doc_ids[dataset_name] = doc_ids
            self.doc_ids = doc_ids

        # تحميل document embeddings
        if dataset_name in EmbeddingQueryProcessor._cache_embeddings:
            self.embeddings = EmbeddingQueryProcessor._cache_embeddings[dataset_name]
        else:
            print(f"[DISK] Loading document embeddings for {dataset_name}")
            embeddings = load_embeddings_joblib(file_prefix)
            EmbeddingQueryProcessor._cache_embeddings[dataset_name] = embeddings
            self.embeddings = embeddings

    def process(self, query_text: str):
        tokens = get_preprocessed_text_terms(query_text, self.dataset_name)
        processed_query_text = " ".join(tokens)
        query_embedding = self.model.encode([processed_query_text])
        return query_embedding, tokens

def process(dataset_name: str, query_text: str):
    qp = EmbeddingQueryProcessor(dataset_name)
    return qp.process(query_text)