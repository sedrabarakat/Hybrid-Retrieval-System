# ranking_emb.py

from sklearn.metrics.pairwise import cosine_similarity
from embedding.embedding_query_processing import EmbeddingQueryProcessor
from collections import OrderedDict

def match_and_rank_embedding(query_text: str, dataset_name: str, similarity_threshold=0.3, top_k=None):
    processor = EmbeddingQueryProcessor(dataset_name)
    query_embedding, tokens = processor.process(query_text)

    embeddings = processor.embeddings
    doc_ids = processor.doc_ids

    similarity_scores = cosine_similarity(query_embedding, embeddings).flatten()

    ranking = {
        doc_id: float(score)
        for doc_id, score in zip(doc_ids, similarity_scores)
        if score >= similarity_threshold
    }

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_ranking = sorted_ranking[:top_k]

    print(f"[EMBEDDING] عرض أعلى {top_k if top_k else 'الكل'} نتائج مرتبة:")
    for rank, (doc_id, score) in enumerate(sorted_ranking[:5], 1):
        print(f"🔹 Rank: {rank}")
        print(f"   Doc ID: {doc_id}")
        print(f"   Score: {score:.6f}")
        print("-" * 50)

    return OrderedDict(sorted_ranking)
