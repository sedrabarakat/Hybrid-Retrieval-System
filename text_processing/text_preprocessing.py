import re
from datetime import datetime
import string
import pycountry
from dateutil import parser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from nltk import pos_tag
from textacy import preprocessing
import spacy

# === Global instances for performance ===
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
country_codes = {c.alpha_3: c.name for c in pycountry.countries}
nlp = spacy.load("en_core_web_sm")

chat_words = {
    "LOL": "laughing out loud",
    "BRB": "be right back",
    "GTG": "got to go",
    "IDK": "i don't know",
    "IMO": "in my opinion",
    "IMHO": "in my humble opinion",
    "FYI": "for your information",
    "TTYL": "talk to you later",
    "ASAP": "as soon as possible",
    "BTW": "by the way",
    "OMG": "oh my god",
    "CU": "see you",
    "U": "you",
    "UR": "your",
    "THX": "thanks",
    "TY": "thank you",
    "YW": "you're welcome",
    "NP": "no problem",
    "ILY": "i love you",
    "JK": "just kidding",
    "IDC": "i don't care",
    "IDK": "i don't know",
    "AFK": "away from keyboard"
}

def _chat_conversion(text):
    new_text = []
    for word in text.split():
        new_text.append(chat_words.get(word.upper(), word))
    return " ".join(new_text)

def _get_words_tokenize(text: str) -> list:
    return tokenizer.tokenize(text)

def _remove_stop_words(tokens: list) -> list:
    return [token for token in tokens if token not in stop_words]

def _stem_tokens(tokens: list) -> list:
    return [stemmer.stem(token) for token in tokens]

def _get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def _lemmatize_tokens(tokens: list) -> list:
    tagged_tokens = pos_tag(tokens)
    return [lemmatizer.lemmatize(token, pos=_get_wordnet_pos(tag)) for token, tag in tagged_tokens]

def get_preprocessed_text_terms(text: str, dataset_name: str) -> list:
    if not text or not text.strip():
        return []

    # Chat word normalization
    text = _chat_conversion(text)

    # Common preprocessing
    text = preprocessing.remove.html_tags(text)
    text = preprocessing.remove.punctuation(text)
    text = preprocessing.remove.brackets(text)
    text = preprocessing.normalize.unicode(text)
    text = preprocessing.normalize.quotation_marks(text)
    text = preprocessing.normalize.hyphenated_words(text)
    text = preprocessing.normalize.whitespace(text)
    text = preprocessing.replace.emojis(text)
    text = preprocessing.replace.urls(text)

    if dataset_name == "beir":
        text = preprocessing.remove.accents(text)

    tokens = _get_words_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = _remove_stop_words(tokens)

    if dataset_name in ["antique", "quora"]:
        tokens = [country_codes.get(t.upper(), t) for t in tokens]

    tokens = _stem_tokens(tokens)
    tokens = _lemmatize_tokens(tokens)

    # Use spaCy for named entity extraction only
    doc = nlp(text)
    named_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    tokens.extend(named_entities)

    return tokens

__all__ = ['get_preprocessed_text_terms']

