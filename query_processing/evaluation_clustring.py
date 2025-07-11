from pathlib import Path
import sys
import os
import time
import json
from typing import Dict, Optional
import mysql.connector
import ir_measures
# تم تصحيح الاستيراد: استخدام R و P مباشرة
from ir_measures import AP, P, R, RR 

# تصحيح مسار الجذر لملف التقييم أيضًا
project_root = Path(__file__).resolve().parents[3] if '__file__' in locals() else Path(os.getcwd()).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# استيراد match_and_rank من المسار الصحيح
from with_clustring.ranking_clustring import match_and_rank
from embedding.embadding_ranking import match_and_rank_embedding # إذا كنت لا تزال تستخدمه


def get_qrels_path(dataset_name: str) -> str:
    paths = {
        "antique": r"C:\Users\HP\IR-project\dataBases\antique_qrels.tsv",
        "beir": r"C:\Users\HP\IR-project\dataBases\beir_qrels.tsv",
    }
    return paths.get(dataset_name, None)

# تم تعديل الدالة لقبول query_id_filter
def get_queries_from_db(dataset_name: str, query_id_filter: Optional[str] = None):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()
    
    query_sql = "SELECT query_id, query_text FROM queries WHERE dataset_name = %s limit 1"
    params = (dataset_name,)

    if query_id_filter:
        query_sql += " AND query_id = %s"
        params += (query_id_filter,)
    
    cursor.execute(query_sql, params)
    rows = cursor.fetchall()
    queries_corpus = {str(row[0]): row[1] for row in rows}
    conn.close()
    return queries_corpus

def get_qrels_from_file(path: str, query_id_filter: Optional[str] = None):
    qrels_corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                query_id, _, doc_id, relevance = parts
                if query_id_filter is None or query_id == query_id_filter:
                    qrels_corpus.append((query_id, doc_id, int(relevance)))
    return qrels_corpus

def build_ground_truth(queries_corpus, qrels_corpus):
    ground_truth = {}
    for query_id in queries_corpus.keys():
        relevant_docs = [
            (doc_id, relevance)
            for q_id, doc_id, relevance in qrels_corpus
            if q_id == query_id
        ]
        ground_truth[query_id] = dict(relevant_docs)
    return ground_truth

def load_doc_id_mapping(dataset_name):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, document_id
        FROM documents
        WHERE dataset_name = %s
    """, (dataset_name,))
    rows = cursor.fetchall()
    conn.close()
    mapping = {int(row['id']): str(row['document_id']) for row in rows}
    return mapping

# تم تعديل دالة get_search_results لقبول معلمة use_clusters
def get_search_results(dataset_name: str, method="tf", use_clusters: bool = False, queries_corpus: Dict[str, str] = None, qrels_query_ids: set = None, doc_id_mapping: Dict[int, str] = None):
    search_results = {}
    
    # إذا لم يتم تمرير corpus/mapping، قم بتحميلها
    if queries_corpus is None:
        queries_corpus = get_queries_from_db(dataset_name)
    if qrels_query_ids is None:
        qrels_corpus_full = get_qrels_from_file(get_qrels_path(dataset_name))
        qrels_query_ids = set(q_id for q_id, _, _ in qrels_corpus_full)
    if doc_id_mapping is None:
        doc_id_mapping = load_doc_id_mapping(dataset_name)

    for query_id, query_text in queries_corpus.items():
        if query_id not in qrels_query_ids:
            print(f"Skipping query {query_id} as it's not in qrels.")
            continue

        print(f"Evaluating query {query_id}: '{query_text[:50]}...'") # طباعة جزء من الاستعلام

        if method == "tf":
            results = match_and_rank(query_text, dataset_name, use_clusters=use_clusters)
        elif method == "embedding":
            results = match_and_rank_embedding(query_text, dataset_name, similarity_threshold=0.3)
        else:
            raise ValueError(f"Unsupported method: {method}")

        converted_results = {}
        for doc_pk, score in results.items():
            doc_real_id = doc_id_mapping.get(int(doc_pk))
            if doc_real_id:
                converted_results[doc_real_id] = score
            else:
                print(f"[!] مفقود mapping لـ doc_id: {doc_pk} (Primary Key).")

        search_results[query_id] = converted_results

    return search_results

# تم تعديل دالة run_evaluation لقبول معلمة use_clusters و single_query_id
def run_evaluation(dataset_name: str, method="tf", use_clusters: bool = False, single_query_id: Optional[str] = None):
    """
    يقوم بتشغيل عملية التقييم لمجموعة بيانات وطريقة معينة، مع إمكانية استخدام تصفية المجموعات
    أو تقييم استعلام واحد محدد.

    Parameters:
    - dataset_name (str): اسم مجموعة البيانات.
    - method (str): طريقة البحث ("tf" أو "embedding").
    - use_clusters (bool): إذا كانت True، سيتم تفعيل منطق تصفية المجموعات في البحث.
    - single_query_id (str, optional): معرف استعلام محدد لتقييمه. إذا تم توفيره،
                                      فسيتم تقييم هذا الاستعلام فقط.
    """
    start_time = time.time() 

    qrels_path = get_qrels_path(dataset_name)
    if qrels_path is None:
        print(f"[!] لا يوجد qrels محدد للداتاسيت: {dataset_name}")
        return

    # جلب الاستعلامات (إما كلها أو استعلام واحد محدد)
    queries_corpus = get_queries_from_db(dataset_name, query_id_filter=single_query_id)
    
    # جلب qrels (إما كلها أو للاستعلام المحدد)
    qrels_corpus = get_qrels_from_file(qrels_path, query_id_filter=single_query_id)
    
    if not queries_corpus:
        print(f"[!] لم يتم العثور على استعلامات للتقييم (ربما query_id '{single_query_id}' غير موجود أو لا يوجد qrels له).")
        return

    ground_truth = build_ground_truth(queries_corpus, qrels_corpus)
    
    # تمرير البيانات التي تم جلبها بالفعل إلى get_search_results لتجنب الجلب المكرر
    qrels_query_ids_set = set(q_id for q_id, _, _ in qrels_corpus)
    doc_id_mapping = load_doc_id_mapping(dataset_name) # لا يمكن تصفية هذه، يجب تحميلها بالكامل

    search_results = get_search_results(
        dataset_name, 
        method=method, 
        use_clusters=use_clusters,
        queries_corpus=queries_corpus, # تمرير الاستعلامات المفلترة
        qrels_query_ids=qrels_query_ids_set, # تمرير qrels_query_ids المفلترة
        doc_id_mapping=doc_id_mapping
    ) 

    if not search_results:
        print("[!] لم يتم الحصول على نتائج بحث لأي استعلام. لا يمكن حساب المقاييس.")
        return

    measures = [AP, R@10, P@10, RR]

    results = ir_measures.calc_aggregate(
        measures,
        ground_truth,
        search_results
    )


    print("\n=== EVALUATION RESULTS ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    elapsed_time = time.time() - start_time 
    print(f"⏱️ زمن التنفيذ الكلي: {elapsed_time:.2f} ثانية") 

    # تعديل اسم ملف الإخراج ليعكس ما إذا كان التقييم لاستعلام واحد
    output_filename_suffix = ""
    if single_query_id:
        output_filename_suffix = f"_query_{single_query_id}"
    if use_clusters:
        output_filename_suffix += "_clustered"
    
    output_path = f"evaluation_results_{method}_{dataset_name}{output_filename_suffix}.json"
    
    results_json = {
        str(metric): value for metric, value in results.items()
    }
    results_json["execution_time_sec"] = round(elapsed_time, 2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)

    print(f"✅ تم حفظ نتائج التقييم في الملف: {output_path}")

# أمثلة على كيفية الاستخدام:
# 1. لتقييم جميع الاستعلامات بدون تجميع
# run_evaluation(dataset_name="beir", method="tf", use_clusters=False)

# 2. لتقييم جميع الاستعلامات مع تجميع
# run_evaluation(dataset_name="beir", method="tf", use_clusters=True)

# 3. لتقييم استعلام واحد محدد (مثلاً، query_id = "1") مع تجميع
# run_evaluation(dataset_name="beir", method="tf", use_clusters=True, single_query_id="1")

# 4. لتقييم استعلام واحد محدد (مثلاً، query_id = "1") بدون تجميع
# run_evaluation(dataset_name="beir", method="tf", use_clusters=False, single_query_id="1")
