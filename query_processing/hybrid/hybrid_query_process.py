import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from query_processing.tf.query_processing import tfidf_query_processor
from query_processing.embedding.embedding_query_processing import EmbeddingQueryProcessor
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix

def process_hybrid_query(query_text: str, dataset_name: str):


    tfidf_processor = tfidf_query_processor(dataset_name)
    tfidf_vector, tokens = tfidf_processor.process(query_text)

    embed_processor = EmbeddingQueryProcessor(dataset_name)
    embedding_vector, _ = embed_processor.process(query_text)

    if embedding_vector.ndim == 1:
        embedding_vector = embedding_vector.reshape(1, -1)

    tfidf_vector = normalize(tfidf_vector)
    embedding_vector = normalize(embedding_vector)

    embedding_sparse = csr_matrix(embedding_vector)
    hybrid_query_vector = hstack([tfidf_vector, embedding_sparse])

    return hybrid_query_vector, tokens
