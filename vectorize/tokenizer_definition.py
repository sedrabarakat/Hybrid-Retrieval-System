from text_processing.text_preprocessing import clean_and_tokenize_text

def tokenizer(text):
    return clean_and_tokenize_text(text, dataset_name="beir")
