
from sentence_transformers import SentenceTransformer
from mysql import connector
import joblib
# ✅ الاتصال بقاعدة البيانات
conn = connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir"
    )
cursor = conn.cursor()

# ✅ جلب الوثائق من العمود processed_text
cursor.execute("SELECT processed_text FROM documents")
rows = cursor.fetchall()

# ✅ استخراج النصوص (الوثائق) من الصفوف
documents = [row[0] for row in rows]

conn.close()

# ✅ تحميل نموذج BERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ توليد Embeddings لكل وثيقة
embeddings = model.encode(documents[0])

# ✅ مثال: طباعة تمثيل أول وثيقة
print(embeddings[0])
joblib.dump(embeddings, 'embeddings.joblib')