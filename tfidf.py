from text_processing.text_preprocessing import get_preprocessed_text_terms

def tokenizer(text):
    return get_preprocessed_text_terms(text, dataset_name="beir")
