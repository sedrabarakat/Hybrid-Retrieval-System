
import sys
import vectorize.tokenizer_definition 

sys.modules["TF_IDF"] = vectorize.tokenizer_definition

from storage.vector_storage import load_vectorizer, load_doc_ids
from text_processing.text_preprocessing import get_preprocessed_text_terms


class QueryProcessor:
    # ✅ الكاش هنا كـ attributes للكلاس
    _cache_vectorizers = {}
    _cache_doc_ids = {}

    def __init__(self, dataset_name: str):
        if dataset_name not in ["antique", "beir"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name
        file_prefix = f"{dataset_name}_all"

        # ✅ نحمّل من الكاش إذا موجود
        if dataset_name in QueryProcessor._cache_vectorizers:
            print(f"[CACHE] Using cached vectorizer for {dataset_name}")

            self.vectorizer = QueryProcessor._cache_vectorizers[dataset_name]
        else:
            print(f"[DISK] Loading vectorizer for {dataset_name}")

            vectorizer = load_vectorizer(file_prefix)

            # تعديل tokenizer
            def tokenizer_no_arg(text):
                return get_preprocessed_text_terms(
                    text,
                    dataset_name=self.dataset_name
                )
            vectorizer.tokenizer = tokenizer_no_arg

            # خزّنه في الكاش
            QueryProcessor._cache_vectorizers[dataset_name] = vectorizer
            self.vectorizer = vectorizer

        # ✅ نحمّل doc_ids من الكاش إذا موجود
        if dataset_name in QueryProcessor._cache_doc_ids:
            self.doc_ids = QueryProcessor._cache_doc_ids[dataset_name]
        else:
            doc_ids = load_doc_ids(file_prefix)
            QueryProcessor._cache_doc_ids[dataset_name] = doc_ids
            self.doc_ids = doc_ids

    def process(self, query_text: str):
        tokens = get_preprocessed_text_terms(query_text, self.dataset_name)
        processed_query_text = " ".join(tokens)
        query_vec = self.vectorizer.transform([processed_query_text])
        return query_vec, tokens


def process(dataset_name: str, query_text: str):
    qp = QueryProcessor(dataset_name)
    return qp.process(query_text)