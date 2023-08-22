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



# Part-of-Speech (POS) Tagging
def pos_tagging(text):
    return pos_tag(word_tokenize(text))


#  Named Entity Recognition (SpaCy)
def named_entity_recognition_spacy(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    return named_entities

# Event Extraction
def extract_events(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    events = []
    current_event = []
    for tag in tagged:
        if tag[1].startswith('VB'):
            current_event.append(tag[0])
        elif current_event:
            events.append(' '.join(current_event))
            current_event = []
    if current_event:
        events.append(' '.join(current_event))
    return events

def specificity_spacy(story):
    sentences = nltk.sent_tokenize(story)
    tagged_sentences = [pos_tagging(sentence) for sentence in sentences]
    named_entities = [named_entity_recognition_spacy(sentence) for sentence in sentences]
    print( "named entities", named_entities)
    events = [extract_events(sentence) for sentence in sentences]
    print("events", events)
    total_named_entities = sum(len(ne_list) for ne_list in named_entities)
    total_events = sum(len(event_list) for event_list in events)
    total_sentences = len(sentences)
    
    threshold = total_sentences // 3
    
    # Check if total_named_entities and total_events are greater than the threshold
    is_specific_story = ( total_named_entities >= threshold ) & ( total_events >= threshold )
    
    return is_specific_story