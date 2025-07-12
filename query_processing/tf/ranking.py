import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from storage.vector_storage import load_tfidf_matrix
from indexing.inverted_index_loader import load_inverted_index
from query_processing.tf.query_processing import tfidf_query_processor
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import numpy as np
from vectorize.tokenizer_definition import tokenizer

def match_and_rank_tfidf(query_text: str, dataset_name: str, top_k=10):
    
    # معالجة الاستعلام
    processor = tfidf_query_processor(dataset_name)
    query_vector, tokens = processor.process(query_text)
    
    # تحميل TF-IDF للوثائق

    tfidf_matrix = load_tfidf_matrix(f"{dataset_name}_all")
    doc_ids = processor.doc_ids

    # حساب التشابه
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # ترتيب النتائج
    ranked_indices = np.argsort(scores)[::-1]
    ranked_results = [(doc_ids[i], float(scores[i])) for i in ranked_indices[:top_k]]

    return OrderedDict(ranked_results)
