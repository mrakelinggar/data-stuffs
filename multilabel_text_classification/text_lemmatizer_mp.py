# text_preprocessing.py
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Initialize lemmatizer and stop words
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



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
    tokens = text.split()
    tagged = pos_tag(tokens)
    lemmatized_sentence = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag) or wordnet.NOUN)
        for word, tag in tagged if word not in stop_words
    ]
    return lemmatized_sentence
