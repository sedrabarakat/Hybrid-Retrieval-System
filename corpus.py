import ir_datasets

def get_corpus(dataset_name: str) -> dict[str, str]:
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
        raise ValueError("Dataset not supported")

    return corpus

# لجلب البيانات من `ir_datasets`
# The file provides a get_corpus() function that loads different datasets based on the input parameter
# - In `indexing.py` it's used to load corpora for building TF-IDF vectorizers
# - In `crawling.py` it provides the base documents for web crawling
# - In `app.py` it's used to get document collections for ranking
# - In `evaluation.py` it helps load test collections for system evaluation