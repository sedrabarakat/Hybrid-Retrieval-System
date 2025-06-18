import shutil
import os
import nltk

# حدد مسار مجلد nltk_data عندك (ممكن تعدله إذا عندك مسار مختلف)
nltk_data_path = os.path.expanduser(r"~\AppData\Roaming\nltk_data")

# المجلدات اللي بدنا نمسحها لإعادة التنزيل
folders_to_remove = [
    'corpora/wordnet',
    'taggers/averaged_perceptron_tagger',
    'tokenizers/punkt',
    'corpora/stopwords'
]

for folder in folders_to_remove:
    full_path = os.path.join(nltk_data_path, folder.replace('/', os.sep))
    if os.path.exists(full_path):
        print(f"Deleting: {full_path}")
        shutil.rmtree(full_path)
    else:
        print(f"Not found, skipping: {full_path}")

# نزل الموارد من جديد
print("\nDownloading NLTK resources:")
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

print("\nDone!")
