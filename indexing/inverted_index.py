import mysql.connector
from collections import defaultdict
import joblib
import os

def build_inverted_index(dataset_name: str):
    # الاتصال بقاعدة البيانات
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
    cursor = conn.cursor()

    # جلب النصوص المعالجة فقط
    cursor.execute("SELECT document_id, processed_text FROM documents WHERE dataset_name = %s", (dataset_name,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    inverted_index = defaultdict(set)

    for doc_id, processed_text in rows:
        # تجاوز المستندات الفارغة
        if not processed_text or not processed_text.strip():
            print(f"[!] الوثيقة {doc_id} تم تجاوزها (النص المعالج فارغ).")
            continue

        terms = processed_text.split()
        if not terms:
            print(f"[!] الوثيقة {doc_id} تم تجاوزها (لا توجد كلمات بعد المعالجة).")
            continue

        for term in set(terms):  # استخدام set لتجنب التكرار داخل المستند
            inverted_index[term].add(str(doc_id))

    # تحويل sets إلى lists
    inverted_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}

    # إنشاء مجلد التخزين إذا لم يكن موجوداً
    output_folder = r"C:\Users\HP\IR-project\indexing\saved_models\inverted_index"
    os.makedirs(output_folder, exist_ok=True)

    # حفظ الفهرس بصيغة joblib
    index_path = os.path.join(output_folder, f"{dataset_name}_inverted_index.joblib")
    joblib.dump(inverted_index, index_path, compress=3)

    print(f"[✓] تم بناء وحفظ الفهرس المعكوس بصيغة joblib لـ: {dataset_name}")
