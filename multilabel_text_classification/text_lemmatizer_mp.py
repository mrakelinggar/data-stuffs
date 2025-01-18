# text_preprocessing.py
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re

# Initialize lemmatizer and stop words
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# List of words to remove
words_to_remove = ["don't", "not", "no", "cannot", "won't", "haven't", "can't", "wasn't", "weren't",
                   "dont", "not", "no", "cannot", "wont", "havent", "cant", "wasnt", "werent", "wouldnt"]

# Remove the words
for word in words_to_remove:
    stop_words.discard(word)  # Discard each word



def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def preprocess_text(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text) 
    tokens_lower = [token.lower() for token in tokens]
    tagged = pos_tag(tokens_lower)
    lemmatized_sentence = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag) or wordnet.NOUN)
        for word, tag in tagged if word not in stop_words
    ]
    return lemmatized_sentence
