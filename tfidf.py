<<<<<<< HEAD
from text_processing.text_preprocessing import get_preprocessed_text_terms

def tokenizer(text):
    return get_preprocessed_text_terms(text, dataset_name="beir")
=======
from text_processing.text_preprocessing import clean_and_tokenize_text

def tokenizer(text):
    return clean_and_tokenize_text(text, dataset_name="beir")
>>>>>>> apis
