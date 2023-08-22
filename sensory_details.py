#imports 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import spacy
from gensim import corpora, models

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def analyze_sensory_details(text):
    sentences = nltk.sent_tokenize(text)

    sensory_keywords = {
        'sight': ['see', 'look', 'watch', 'observe', 'gaze', 'view', 'stare', 'sight'],
        'sound': ['hear', 'listen', 'sound', 'noise', 'auditory', 'echo', 'sound'],
        'smell': ['smell', 'scent', 'aroma', 'fragrance', 'odor', 'perfume', 'smell'],
        'taste': ['taste', 'flavor', 'savor', 'palate', 'tasty', 'delicious', 'taste'],
        'touch': ['feel', 'touch', 'texture', 'tactile', 'surface', 'contact', 'touch']
    }

    sensory_details = []
    for sentence in sentences:
        doc = nlp(sentence)
        lemmatized_words = [token.lemma_ for token in doc]

        for sense, keywords in sensory_keywords.items():
            for keyword in keywords:
                if keyword in lemmatized_words:
                    sensory_details.append({'sense': sense, 'sentence': sentence})
                    break

    return sensory_details



# Example usage
text = "I saw a beautiful sunset and heard the sound of waves crashing. I don't see well"
sensory_results = analyze_sensory_details(text)
print(sensory_results)


# In[ ]:


def has_sensory_details(text):
    sensory_details = analyze_sensory_details(text)
    return "yes" if sensory_details else "no"

text = "The sun sets over the mountains, casting a warm orange glow. Birds sing in the trees, and the scent of blooming flowers fills the air."
result = has_sensory_details(text)
print(result)