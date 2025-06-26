

import sys
import vectorize.tokenizer_definition 

sys.modules["TF_IDF"] = vectorize.tokenizer_definition

from storage.vector_storage import load_vectorizer, load_doc_ids
from text_processing.text_preprocessing import get_preprocessed_text_terms

class QueryProcessor:
    def __init__(self, dataset_name: str):
        if dataset_name not in ["antique", "beir"]:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.dataset_name = dataset_name
        file_prefix = f"{dataset_name}_all"

        # تحميل vectorizer
        self.vectorizer = load_vectorizer(file_prefix)

        # تعديل دالة tokenizer داخل vectorizer لتدعو دالة المعالجة المسبقة مع dataset_name
        def tokenizer_no_arg(text):
            return get_preprocessed_text_terms(text, dataset_name=self.dataset_name)

        self.vectorizer.tokenizer = tokenizer_no_arg

        # تحميل معرّفات المستندات
        self.doc_ids = load_doc_ids(file_prefix)

    def process(self, query_text: str):
        tokens = get_preprocessed_text_terms(query_text, self.dataset_name)
        processed_query_text = " ".join(tokens)
        query_vec = self.vectorizer.transform([processed_query_text])
        return query_vec, tokens
def process(dataset_name: str, query_text: str):
    qp = QueryProcessor(dataset_name)
    return qp.process(query_text)
