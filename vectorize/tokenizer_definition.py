from text_processing.text_preprocessing import get_preprocessed_text_terms

def tokenizer(text):
    # هنا ثبت dataset_name مثلا "antique" أو غيره، حسب ما تم بناء النموذج
    return get_preprocessed_text_terms(text, dataset_name="beir")
