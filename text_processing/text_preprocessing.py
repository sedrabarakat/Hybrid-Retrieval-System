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
import unicodedata

# === Global instances for performance ===
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
country_codes = {c.alpha_3: c.name for c in pycountry.countries}
nlp = spacy.load("en_core_web_sm")

chat_words = {
    "LOL": "laughing out loud", "BRB": "be right back", "GTG": "got to go",
    "IDK": "i don't know", "IMO": "in my opinion", "IMHO": "in my humble opinion",
    "FYI": "for your information", "TTYL": "talk to you later", "ASAP": "as soon as possible",
    "BTW": "by the way", "OMG": "oh my god", "CU": "see you", "U": "you", "UR": "your",
    "THX": "thanks", "TY": "thank you", "YW": "you're welcome", "NP": "no problem",
    "ILY": "i love you", "JK": "just kidding", "IDC": "i don't care", "AFK": "away from keyboard"
}

def _chat_conversion(text: str) -> str:
    return " ".join([chat_words.get(w.upper(), w) for w in text.split()])

def _get_words_tokenize(text: str) -> list:
    return tokenizer.tokenize(text)

def _remove_stop_words(tokens: list) -> list:
    return [token for token in tokens if token not in stop_words]

def _stem_tokens(tokens: list) -> list:
    return [stemmer.stem(token) for token in tokens]

def _get_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return "a"
    elif tag.startswith("V"):
        return "v"
    elif tag.startswith("N"):
        return "n"
    elif tag.startswith("R"):
        return "r"
    else:
        return "n"


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




def clean_and_tokenize_text(text: str, dataset_name: str = "beir", is_query: bool = False) -> list:
    if not text or not text.strip():
        return []

    text = _chat_conversion(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[\r\n\t]", " ", text)
    text = re.sub(r"\b([a-zA-Z]+)-([a-zA-Z]+)\b", r"\1\2", text)  # Merge hyphenated words
    text = re.sub(r"\s+", " ", text).strip()

    if dataset_name.lower() == "beir":
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    text = re.sub(rf"[{re.escape(string.punctuation.replace('?', '').replace('!', ''))}]", "", text)

    if is_query:
        text = text.replace("?", "").replace("!", "")

    tokens = tokenizer.tokenize(text.lower())

    important_stops = {"not", "is", "are", "do", "does", "did", "what", "who", "how", "why", "when", "where",
                       "could", "should", "would", "can", "may", "also", "must", "still"}
    tokens = [t for t in tokens if t not in stop_words or t in important_stops]

    tokens = [t for t in tokens if not t.isdigit()]
    tokens = [t for t in tokens if len(t) > 2 or t in important_stops]

    tagged_tokens = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(t, _get_wordnet_pos(tag)) for t, tag in tagged_tokens]

    return tokens


