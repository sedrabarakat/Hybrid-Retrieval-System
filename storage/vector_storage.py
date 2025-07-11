import os
import joblib
import scipy.sparse
import numpy as np

# المسار الأساسي
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorize", "saved_models"))

# إنشاء المجلد تلقائياً حسب نوع الـ vectorizer
def _get_dir(vectorizer_type: str) -> str:
    dir_path = os.path.join(BASE_PATH, vectorizer_type)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# حفظ الـ vectorizer
def save_vectorizer(model, dataset_name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(_get_dir(vectorizer_type), f"{dataset_name}_tfidf_vectorizer.joblib")
    joblib.dump(model, path, compress=3)


def load_vectorizer(dataset_name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{dataset_name}_tfidf_vectorizer.joblib")
    model = joblib.load(path)
    
    return model

# حفظ مصفوفة TF-IDF بصيغة sparse
def save_tfidf_matrix(matrix, name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(_get_dir(vectorizer_type), f"{name}_tfidf_matrix.npz")
    scipy.sparse.save_npz(path, matrix)

# تحميل مصفوفة TF-IDF
def load_tfidf_matrix(name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_tfidf_matrix.npz")
    return scipy.sparse.load_npz(path)

# (اختياري) حفظ Embedding بصيغة numpy array
def save_embeddings(array, name: str, vectorizer_type: str = "embedding"):
    path = os.path.join(_get_dir(vectorizer_type), f"{name}_embeddings.npy")
    np.save(path, array)

# تحميل Embedding
def load_embeddings(name: str, vectorizer_type: str = "embedding"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_embeddings.npy")
    return np.load(path)
    
def load_embeddings_joblib(name: str, vectorizer_type: str = "embedding"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_embeddings.joblib")
    return joblib.load(path)


def save_hybrid(array, name: str, vectorizer_type: str = "Hybrid"):
    path = os.path.join(_get_dir(vectorizer_type), f"{name}_hybrid.joblib")
    joblib.dump(array, path)

def load_hybrid(name: str, vectorizer_type: str = "Hybrid"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_hybrid.joblib")
    return joblib.load(path)


# حفظ hybrid
def save_hybrid(array, name: str, vectorizer_type: str = "Hybrid"):
    path = os.path.join(_get_dir(vectorizer_type), f"{name}_hybrid.joblib")
    joblib.dump(array, path)


def load_embeddings_ids(name: str, vectorizer_type: str = "embedding"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_embeddings_doc_ids.joblib")
    return joblib.load(path)


def load_tfidf_ids(dataset_name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{dataset_name}_tfidf_all_doc_ids.joblib")
    return joblib.load(path)

def save_tfidf_doc_ids(dataset_name: str, doc_ids: list):
    path = os.path.join(BASE_PATH, f"{dataset_name}_tfidf_all_doc_ids.joblib")
    joblib.dump(doc_ids, path)


def load_hyprid_ids(dataset_name: str, vectorizer_type: str = "hybrid"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{dataset_name}_hybrid_all_docs_ids.joblib")
    return joblib.load(path)


def get_qrels_file_path(dataset_name: str) -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(base_dir, "dataBases")
    file_name = f"{dataset_name}_qrels.tsv"
    full_path = os.path.join(data_folder, file_name)
    return full_path



############


# حفظ معرّفات المستندات
def save_doc_ids(doc_ids, file_suffix, vectorizer_type="tfidf"):
    path = os.path.join(_get_dir(vectorizer_type), f"{file_suffix}_doc_ids.joblib")
    joblib.dump(doc_ids, path)


def load_doc_ids(file_suffix, vectorizer_type="tfidf"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{file_suffix}_doc_ids.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document IDs file not found at: {path}")
    doc_ids = joblib.load(path)

    # إرجاع القائمة كما هي بدون تحويل إلى سترينغ
    return doc_ids

def load_tfidf_vectorizer(name):
    path = os.path.join(
        "c:\\Users\\HP\\IR-project\\vectorize\\saved_models\\tfidf",
        f"{name}_vectorizer.joblib"
    )
    
    vectorizer = joblib.load(path)
    
    return vectorizer