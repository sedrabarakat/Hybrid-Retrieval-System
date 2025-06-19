# import os
# import joblib
# import scipy.sparse

# BASE_PATH = "saved_models"
# os.makedirs(BASE_PATH, exist_ok=True)

# # حفظ النموذج (TF-IDF vectorizer أو BM25 أو أي model)
# def save_vectorizer(model, name):
#     path = os.path.join(BASE_PATH, f"{name}_vectorizer.joblib")
#     joblib.dump(model, path, compress=3)
#     print(f"[✓] Model saved to: {path}")

# # تحميل النموذج
# def load_vectorizer(name):
#     path = os.path.join(BASE_PATH, f"{name}_vectorizer.joblib")
#     return joblib.load(path)

# # حفظ مصفوفة TF-IDF (sparse matrix)
# def save_tfidf_matrix(matrix, name):
#     path = os.path.join(BASE_PATH, f"{name}_tfidf.npz")
#     scipy.sparse.save_npz(path, matrix)
#     print(f"[✓] TF-IDF Matrix saved to: {path}")

# # تحميل مصفوفة TF-IDF
# def load_tfidf_matrix(name):
#     path = os.path.join(BASE_PATH, f"{name}_tfidf.npz")
#     return scipy.sparse.load_npz(path)
# storge/vector_storge.py

import os
import joblib
import scipy.sparse

BASE_PATH = "saved_models"
os.makedirs(BASE_PATH, exist_ok=True)

def save_vectorizer(model, name):
    path = os.path.join(BASE_PATH, f"{name}_vectorizer.joblib")
    joblib.dump(model, path, compress=3)

def load_vectorizer(name):
    path = os.path.join(BASE_PATH, f"{name}_vectorizer.joblib")
    return joblib.load(path)

def save_tfidf_matrix(matrix, name):
    path = os.path.join(BASE_PATH, f"{name}_tfidf.npz")
    scipy.sparse.save_npz(path, matrix)

def load_tfidf_matrix(name):
    path = os.path.join(BASE_PATH, f"{name}_tfidf.npz")
    return scipy.sparse.load_npz(path)
