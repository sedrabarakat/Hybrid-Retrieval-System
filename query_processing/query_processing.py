import joblib
import os
from text_processing.text_preprocessing import get_preprocessed_text_terms

class QueryProcessor:
    def __init__(self, dataset_name):
        model_dir = os.path.join("vectorize", "saved_models", "tfidf")
        self.vectorizer = joblib.load(os.path.join(model_dir, f"{dataset_name}_all_vectorizer.joblib"))
        self.doc_ids = joblib.load(os.path.join(model_dir, f"{dataset_name}_all_doc_ids.joblib"))
        self.dataset_name = dataset_name
    
    def process(self, query_text):
        # 1. تنظيف ومعالجة الاستعلام (إرجاع قائمة tokens)
        tokens = get_preprocessed_text_terms(query_text, self.dataset_name)
        
        # 2. إعادة تجميع التوكينز كنص مفصول بمسافات
        processed_query_text = " ".join(tokens)
        
        # 3. تحويل النص المعالج إلى تمثيل TF-IDF vector
        query_vec = self.vectorizer.transform([processed_query_text])
        return query_vec
