import os
import joblib
import scipy.sparse
import numpy as np

# المسار الأساسي
BASE_PATH = "saved_models"

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
    return joblib.load(path)

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
