import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing.text_preprocessing import get_preprocessed_text_terms
import storage  # تأكد من وجود هذه الوحدة وتعمل حفظ

# تابع tokenizer مخصص يطبق get_preprocessed_text_terms
def custom_tokenizer(text: str, dataset_name: str):
    return get_preprocessed_text_terms(text, dataset_name)

def build_save_vectorizer(dataset_name: str):
    # الاتصال بقاعدة البيانات
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()

    # جلب النصوص من قاعدة البيانات
    cursor.execute("SELECT text FROM documents WHERE dataset_name = %s", (dataset_name,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    raw_texts = [row[0] for row in rows]

    # تعريف TfidfVectorizer مع توكنزر مخصص
    vectorizer = TfidfVectorizer(
        tokenizer=lambda text: custom_tokenizer(text, dataset_name),
        lowercase=False,
        preprocessor=None,
        token_pattern=None  # ضروري جدًا لتعطيل regex الافتراضي
    )

    # تدريب الـ TF-IDF على النصوص الخام (سيتم تمريرها لتوكنزرنا المخصص تلقائيًا)
    tfidf_matrix = vectorizer.fit_transform(raw_texts)

    # حفظ النتائج
    storage.save_vectorizer(vectorizer, dataset_name)
    storage.save_tfidf_matrix(tfidf_matrix, dataset_name)

    print(f"[✓] Vectorizer and matrix for '{dataset_name}' saved successfully.")
