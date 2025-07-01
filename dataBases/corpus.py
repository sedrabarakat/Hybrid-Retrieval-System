import ir_datasets
from collections import defaultdict

def get_corpus(dataset_name: str) -> dict[str, str]:
    """
    تحميل مستندات أو استعلامات من ir_datasets حسب dataset_name.
    - 'beir' أو 'antique' لتحميل المستندات (docs)
    - 'beir-queries' أو 'antique-queries' لتحميل الاستعلامات (queries)
    """
    if dataset_name == "beir":
        dataset = ir_datasets.load("beir/webis-touche2020/v2")
        corpus = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

    elif dataset_name == "antique":
        dataset = ir_datasets.load("antique/train")
        corpus = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

    elif dataset_name == "beir-queries":
        dataset = ir_datasets.load("beir/webis-touche2020/v2")
        corpus = {query.query_id: query.text for query in dataset.queries_iter()}

    elif dataset_name == "antique-queries":
        dataset = ir_datasets.load("antique/train")
        corpus = {query.query_id: query.text for query in dataset.queries_iter()}

    else:
        raise ValueError(f"❌ Dataset غير مدعوم: {dataset_name}")

    return corpus


def get_qrels(dataset_name: str) -> dict[str, dict[str, int]]:
    """
    تحميل تقييمات الصلاحية (qrels) من ir_datasets بصيغة:
    dict[query_id][doc_id] = relevance_score
    """
    if dataset_name == "beir":
        dataset = ir_datasets.load("beir/webis-touche2020/v2")
        qrels_raw = dataset.qrels_iter()

    elif dataset_name == "antique":
        dataset = ir_datasets.load("antique/train")
        qrels_raw = dataset.qrels_iter()

    else:
        raise ValueError(f"❌ qrels غير مدعومة لهذه الداتا سيت: {dataset_name}")

    qrels_dict = defaultdict(dict)
    for item in qrels_raw:
        qrels_dict[str(item.query_id)][str(item.doc_id)] = int(item.relevance)

    return qrels_dict


def save_qrels_to_tsv(qrels: dict[str, dict[str, int]], filepath: str):
    """
    حفظ qrels في ملف TSV بصيغة:
    query_id \t 0 \t doc_id \t relevance
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for query_id, docs in qrels.items():
            for doc_id, relevance in docs.items():
                f.write(f"{query_id}\t0\t{doc_id}\t{relevance}\n")
    print(f"✅ تم حفظ qrels في الملف: {filepath}")
