
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from storage.vector_storage import load_vectorizer, load_tfidf_ids
from text_processing.text_preprocessing import clean_and_tokenize_text
from vectorize.tokenizer_definition import tokenizer
from text_processing.text_preprocessing import clean_and_tokenize_text


class tfidf_query_processor:
    _cache_vectorizers = {}
    _cache_doc_ids = {}

    def __init__(self, dataset_name: str):
        if dataset_name not in ["antique", "beir"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name

         # Load vectorizer
        if dataset_name in tfidf_query_processor._cache_vectorizers:
            self.vectorizer = tfidf_query_processor._cache_vectorizers[dataset_name]
        else:
            vectorizer = load_vectorizer(dataset_name=dataset_name)

            #tokenizer
            def tokenizer_no_arg(text):
                return clean_and_tokenize_text(
                    text,
                    dataset_name=self.dataset_name
                )
            vectorizer.tokenizer = tokenizer_no_arg

            # caching
            tfidf_query_processor._cache_vectorizers[dataset_name] = vectorizer
            self.vectorizer = vectorizer

        # loading ids from cache
        if dataset_name in tfidf_query_processor._cache_doc_ids:
            self.doc_ids = tfidf_query_processor._cache_doc_ids[dataset_name]
        else:
            doc_ids = load_tfidf_ids(dataset_name=dataset_name)
            tfidf_query_processor._cache_doc_ids[dataset_name] = doc_ids
            self.doc_ids = doc_ids

    def process(self, query_text: str):
        tokens = clean_and_tokenize_text(query_text, self.dataset_name)
        processed_query_text = " ".join(tokens)
        query_vec = self.vectorizer.transform([processed_query_text])
        return query_vec, tokens


def process(dataset_name: str, query_text: str):
    
    qp = tfidf_query_processor(dataset_name)
    return qp.process(query_text)


