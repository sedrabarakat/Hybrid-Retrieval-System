import os
import joblib

def load_inverted_index(dataset_name: str):
    # المسار الصريح إلى مجلد الفهارس المعكوسة
    # base_path = r"C:\Users\HP\IR-project\indexing\saved_models\inverted_index"
    base_path = r"F:\IR\IR\indexing\saved_models\inverted_index"
    
    
    # اسم الملف حسب اسم الداتاست
    filename = f"{dataset_name}_inverted_index.joblib"
    
    # المسار الكامل
    path = os.path.join(base_path, filename)

    # طباعة للمراجعة
    print("[*] محاولة تحميل الفهرس من:", path)

    # التحقق من وجود الملف
    if not os.path.exists(path):
        raise FileNotFoundError(f"لم يتم العثور على الفهرس المعكوس: {path}")
    
    return joblib.load(path)
