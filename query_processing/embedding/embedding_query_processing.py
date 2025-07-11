# embedding_query_processing.py

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from sentence_transformers import SentenceTransformer
from text_processing.text_preprocessing import clean_and_tokenize_text
import sklearn
from storage.vector_storage import load_embeddings_joblib, load_embeddings_ids

    
class EmbeddingQueryProcessor:
    _cache_models = {}
    _cache_doc_ids = {}
    _cache_embeddings = {}

    def __init__(self, dataset_name: str, model_name: str = "all-mpnet-base-v2"):
        if dataset_name not in ["antique", "beir"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.model_name = model_name
        file_prefix = f"{dataset_name}_all"

        if model_name in EmbeddingQueryProcessor._cache_models:
            self.model = EmbeddingQueryProcessor._cache_models[model_name]
        else:
            model = SentenceTransformer(model_name)
            EmbeddingQueryProcessor._cache_models[model_name] = model
            self.model = model

        if dataset_name in EmbeddingQueryProcessor._cache_doc_ids:
            self.doc_ids = EmbeddingQueryProcessor._cache_doc_ids[dataset_name]
        else:
            doc_ids = load_embeddings_ids(name=file_prefix)
            EmbeddingQueryProcessor._cache_doc_ids[dataset_name] = doc_ids
            self.doc_ids = doc_ids


        if dataset_name in EmbeddingQueryProcessor._cache_embeddings:
            self.embeddings = EmbeddingQueryProcessor._cache_embeddings[dataset_name]
        else:
            print(f"[LOAD] تحميل document embeddings: {dataset_name}")
            embeddings = load_embeddings_joblib(name=file_prefix)
            embeddings = sklearn.preprocessing.normalize(embeddings, norm='l2', axis=1)

            EmbeddingQueryProcessor._cache_embeddings[dataset_name] = embeddings
            self.embeddings = embeddings

    def process(self, query_text: str):
        cleaned = clean_and_tokenize_text(query_text, self.dataset_name, is_query=True)

        query_embedding = self.model.encode(
          [" ".join(cleaned)], 
          #device=self.device,
          normalize_embeddings=True,
          batch_size=16,
          show_progress_bar=False
    )
        return query_embedding, cleaned


def process(dataset_name: str, query_text: str):
    qp = EmbeddingQueryProcessor(dataset_name)
    return qp.process(query_text)

