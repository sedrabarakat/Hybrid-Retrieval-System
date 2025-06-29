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
def save_vectorizer(model, name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(_get_dir(vectorizer_type), f"{name}_vectorizer.joblib")
    joblib.dump(model, path, compress=3)

# تحميل الـ vectorizer
def load_vectorizer(name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_vectorizer.joblib")
    
    model = joblib.load(path)
    
    return model

# حفظ مصفوفة TF-IDF بصيغة sparse
def save_tfidf_matrix(matrix, name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(_get_dir(vectorizer_type), f"{name}_tfidf.npz")
    scipy.sparse.save_npz(path, matrix)

# تحميل مصفوفة TF-IDF
def load_tfidf_matrix(name: str, vectorizer_type: str = "tfidf"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_tfidf.npz")
    return scipy.sparse.load_npz(path)

# (اختياري) حفظ Embedding بصيغة numpy array
def save_embeddings(array, name: str, vectorizer_type: str = "embedding"):
    path = os.path.join(_get_dir(vectorizer_type), f"{name}_embeddings.npy")
    np.save(path, array)

# تحميل Embedding
def load_embeddings(name: str, vectorizer_type: str = "embedding"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_embeddings.npy")
    return np.load(path)

# حفظ hybrid
def save_hybrid(array, name: str, vectorizer_type: str = "Hybrid"):
    path = os.path.join(_get_dir(vectorizer_type), f"{name}_hybrid.joblib")
    joblib.dump(array, path)

# تحميل hybrid
def load_hybrid(name: str, vectorizer_type: str = "Hybrid"):
    path = os.path.join(BASE_PATH, vectorizer_type, f"{name}_hybrid.joblib")
    return joblib.load(path)

# حفظ معرّفات المستندات
def save_doc_ids(doc_ids, file_suffix, vectorizer_type="tfidf"):
    path = os.path.join(_get_dir(vectorizer_type), f"{file_suffix}_doc_ids.joblib")
    joblib.dump(doc_ids, path)

# # تحميل معرّفات المستندات
# def load_doc_ids(file_suffix, vectorizer_type="tfidf"):
#     path = os.path.join(BASE_PATH, vectorizer_type, f"{file_suffix}_doc_ids.joblib")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Document IDs file not found at: {path}")
#     return joblib.load(path)

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