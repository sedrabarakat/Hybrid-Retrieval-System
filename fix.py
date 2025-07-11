import shutil
import os
import nltk

# مسار مجلد nltk_data
nltk_data_path = os.path.expanduser(r"~\AppData\Roaming\nltk_data")

# الموارد التي سيتم حذفها
folders_to_remove = [
    'corpora/wordnet',
    'corpora/omw-1.4',
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

# التأكد أن nltk سيستخدم هذا المسار
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# إعادة تنزيل الموارد
print("\nDownloading NLTK resources:")
for pkg in ['wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'punkt', 'stopwords']:
    nltk.download(pkg, download_dir=nltk_data_path)

print("\nDone!")
