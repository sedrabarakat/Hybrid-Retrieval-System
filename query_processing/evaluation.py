import pandas as pd
from sklearn.metrics import average_precision_score

def compute_map(ranked_docs_dict, qrels_path):
    """
    حساب Mean Average Precision (MAP) لاستعلام واحد.

    ranked_docs_dict: OrderedDict أو dict يحتوي على {doc_id: score} مرتبة تنازليًا
    qrels_path: مسار ملف تقييم الوثائق qrels بتنسيق TSV، بأعمدة: query_id, doc_id, relevance_score
                (يمكن أن يحتوي على أكثر من استعلام لكننا نفترض استعلام واحد فقط)
    """

    # قراءة ملف qrels
    qrels_df = pd.read_csv(qrels_path, sep='\t', names=["query_id", "doc_id", "score"])

    # نعتبر الوثائق ذات score > 0 هي ذات صلة
    relevant_docs = set(qrels_df[qrels_df["score"] > 0]["doc_id"])

    y_true = []
    y_scores = []

    for doc_id, score in ranked_docs_dict.items():
        y_true.append(1 if doc_id in relevant_docs else 0)
        y_scores.append(score)

    if not any(y_true):
        # لا توجد وثائق ذات صلة
        return 0.0

    return average_precision_score(y_true, y_scores)
