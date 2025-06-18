import re
from datetime import datetime

import string
import pycountry
from nltk import pos_tag
from dateutil import parser
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer


from textacy import preprocessing


def get_preprocessed_text_terms(text: str, dataset_name: str) -> list:
    if(dataset_name == "antique" or dataset_name == "antique-queries"):
        text = preprocessing.remove.html_tags(text)
        text = preprocessing.remove.punctuation(text)
        text = preprocessing.remove.brackets(text)

        text = preprocessing.normalize.unicode(text)
        text = preprocessing.normalize.quotation_marks(text)
        text = preprocessing.normalize.hyphenated_words(text)
        text = preprocessing.normalize.whitespace(text)

        text = preprocessing.replace.emojis(text)
        text = preprocessing.replace.urls(text)
        # text = preprocessing.remove.accents(text)
        # text = _chat_conversion(text)

        tokens = _get_words_tokenize(text)
        lowercase_tokens = _lowercase_tokens(tokens)

        filtered_tokens = _remove_stop_words(lowercase_tokens)
        # d = _normalize_dates(filtered_tokens)
        c = _normalize_country_names(filtered_tokens)

        stemmed_tokens = _stem_tokens(c)
        lemmitized_tokens = _lemmatize_tokens(stemmed_tokens)
        return lemmitized_tokens

    elif dataset_name == "beir" :
        text = preprocessing.remove.html_tags(text)
        text = preprocessing.remove.punctuation(text)
        text = preprocessing.remove.brackets(text)

        text = preprocessing.normalize.unicode(text)
        text = preprocessing.normalize.quotation_marks(text)
        text = preprocessing.normalize.hyphenated_words(text)
        text = preprocessing.normalize.whitespace(text)

        text = preprocessing.replace.emojis(text)
        text = preprocessing.replace.urls(text)
        text = preprocessing.remove.accents(text)
        # text = _chat_conversion(text)

        tokens = _get_words_tokenize(text)
        lowercase_tokens = _lowercase_tokens(tokens)

        filtered_tokens = _remove_stop_words(lowercase_tokens)
        # d = _normalize_dates(filtered_tokens)
        # c = _normalize_country_names(filtered_tokens)

        stemmed_tokens = _stem_tokens(filtered_tokens)
        lemmitized_tokens = _lemmatize_tokens(stemmed_tokens)
        return lemmitized_tokens

    elif dataset_name == "quora":
        text = preprocessing.remove.html_tags(text)
        text = preprocessing.remove.punctuation(text)
        text = preprocessing.remove.brackets(text)

        text = preprocessing.normalize.unicode(text)
        text = preprocessing.normalize.quotation_marks(text)
        text = preprocessing.normalize.hyphenated_words(text)
        text = preprocessing.normalize.whitespace(text)

        text = preprocessing.replace.emojis(text)
        text = preprocessing.replace.urls(text)
        # text = preprocessing.remove.accents(text)
        # text = _chat_conversion(text)

        tokens = _get_words_tokenize(text)
        lowercase_tokens = _lowercase_tokens(tokens)

        filtered_tokens = _remove_stop_words(lowercase_tokens)
        # d = _normalize_dates(filtered_tokens)
        c = _normalize_country_names(filtered_tokens)

        stemmed_tokens = _stem_tokens(c)
        lemmitized_tokens = _lemmatize_tokens(stemmed_tokens)
        return lemmitized_tokens  

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

def _get_words_tokenize(text: str) -> list:
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)


def _lowercase_tokens(tokens: list) -> list:
    return [token.lower() for token in tokens]


def _remove_stop_words(tokens: list) -> list:
    # question_words = {'what', 'who', 'whom', 'whose', 'which', 'when', 'where', 'why', 'how', 'how much', 'how many',
    #                 'how long', 'how often', 'how far', 'how old', 'how come'}
    # stop_words = set(stopwords.words('english')) - question_words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens


def _stem_tokens(tokens: list) -> list:
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def _normalize_dates(tokens: list):
    normalized_tokens = []
    for word in tokens:
        new_text = word
        try:
            dt = parser.parse(word)
            if isinstance(dt, datetime):
                date_obj = parser.parse(word)
                formatted_date = date_obj.strftime("%Y %m %d")
                day_name = date_obj.strftime("%A")
                month_name = date_obj.strftime("%B")
                time_obj = date_obj.time().strftime("%I %M %p")
                new_formatted = f"{formatted_date} {day_name} {month_name} {time_obj}"
                new_text = new_text.replace(word, new_formatted)
        except (ValueError, OverflowError):
            pass
        normalized_tokens.append(new_text)
    return normalized_tokens


def _normalize_country_names(tokens: list) -> list:
    # Create a set of country names for faster lookup
    # ex. {'USA', 'KSA'}
    country_codes = set(country.alpha_3 for country in pycountry.countries)
    # Loop over the tokens and update country names if they match a country name
    for token in tokens.copy():
        if token.upper() in country_codes:
            try:
                country = pycountry.countries.lookup(token.upper())
                tokens.remove(token)
                tokens.append(country.name) # 'USA' -> 'United States of America'
            except LookupError:
                pass

    # Return the updated list of tokens
    return tokens


def _get_wordnet_pos(treebank_tag):
    """
    Convert the Penn Treebank POS tags to WordNet POS tags.
    """
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Use 'n' (noun) as default if no match is found


def _lemmatize_tokens(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    # فقط استعمل pos_tag بدون باراميترات اضافية
    tagged_tokens = pos_tag(tokens)  # بدون lang أو غيره
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos=_get_wordnet_pos(pos_tag)) for token, pos_tag in tagged_tokens]
    return lemmatized_tokens


def _chat_conversion(text):
    new_text = []
    for i in text.split():
        if i.upper() in chat_words:
            new_text.append(chat_words[i.upper()])
        else:
            new_text.append(i)
    return " ".join(new_text)

__all__ = ['get_preprocessed_text_terms', '_get_words_tokenize']


chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}
