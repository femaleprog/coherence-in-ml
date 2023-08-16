# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


# %%
data = pd.read_csv('/kaggle/input/dataset0/data.csv')

# %% [markdown]
#  Criteria for defining a personal event memory by Pillemer ( 1998 ) :  
# *  (a) present a specific event that took place at a particular time and place, rather than a summary event or extended series of events.
# * (b) contain a detailed account of the rememberer's own personal circumstances at the time of the event. 
# * (c) evoke sensory images or bodily sensations that contribute to the feeling of "re-experiencing" or "reliving" the event.
# * (d) link its details and images to a particular moment or moments of phenomenal experience. 
# * (e) be believed to be a truthful representation of what actually transpired.

# %% [markdown]
# ## 1. Specificity

# %%
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Tokenization
def tokenize_text(text):
    return sent_tokenize(text)

# Part-of-Speech (POS) Tagging
def pos_tagging(text):
    tokens = word_tokenize(text)
    return pos_tag(tokens)


#  Named Entity Recognition (NER)
def named_entity_recognition(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    ne_tree = ne_chunk(tagged)
    named_entities = []
    for chunk in ne_tree:
        if hasattr(chunk, 'label') and chunk.label() == 'NE':
            named_entities.append(' '.join(c[0] for c in chunk))
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





# %% [markdown]
# ### Testing function for specificity

# %%
print(data.iloc[0]['Story'])

# %%
def specificity_ne(story):
    sentences = tokenize_text(story)
    print('Sentences:', sentences)
    print()
    tagged_sentences = [pos_tagging(sentence) for sentence in sentences]
    print('Tagged Sentences:', tagged_sentences)
    print()
    named_entities = [named_entity_recognition(sentence) for sentence in sentences]
    print('Named Entities:', named_entities)
    print()
    events = [extract_events(sentence) for sentence in sentences]
    print('Events:', events)
    print()


# %% [markdown]
# Testing

# %%
specificity_ne(data.iloc[0]['Story'])

# %% [markdown]
# The model is not able to capture named entities. Let's try with another example  

# %%
specificity_ne(data.iloc[1]['Story'])

# %% [markdown]
#  NE was not able to capture any named entity. Let's try with other python library. 
#  Let's start with SpaCy

# %%
import spacy

def named_entity_recognition_spacy(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    return named_entities


# %%
named_entities = [named_entity_recognition_spacy(sentence) for sentence in sentences]
print('Named Entities:', named_entities)

# %%
def specificity_ne_spacy(story):
    sentences = tokenize_text(story)
    print('Sentences:', sentences)
    tagged_sentences = [pos_tagging(sentence) for sentence in sentences]
    print('Tagged Sentences:', tagged_sentences)
    named_entities = [named_entity_recognition_spacy(sentence) for sentence in sentences]
    print('Named Entities:', named_entities)
    events = [extract_events(sentence) for sentence in sentences]
    print('Events:', events)


# %% [markdown]
# Testing the specifity_ne_spacy function :

# %%
specificity_ne_spacy(data.iloc[0]['Story'])

# %%
specificity_ne_spacy(data.iloc[1]['Story'])

# %% [markdown]
# # 2.Personal Context 

# %%
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim import corpora, models

def evaluate_personal_context(text):
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(sentence)["compound"] for sentence in sentences]

    # Topic Modeling
    tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_sentences]
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary)

    # Extract most significant topics
    topics = [lda_model.get_document_topics(doc) for doc in corpus]
    most_significant_topics = [max(topic, key=lambda x: x[1]) for topic in topics]

    # Get actual topics
    actual_topics = [lda_model.print_topic(topic[0]) for topic in most_significant_topics]

    # Return sentiment scores and actual topics
    return sentiment_scores, actual_topics


# %% [markdown]
# In the context of topic modeling with LDA, the weights assigned to each word in a topic represent the importance or prevalence of that word within the topic. In the output you provided
# 
# Understanding the significance of individual words and their weights within a topic can help provide insights into the key themes and subjects present in the text.

# %%
def test_personal_context(text):
    sentiment_scores, actual_topics = evaluate_personal_context(text)
    print("Sentiment Scores:", sentiment_scores)
    print()
    print("Actual Topics:")
    for topic in actual_topics:
        print(topic)

# %%
test_personal_context(data.iloc[0]['Story'])

# %%
test_personal_context("i lost my best friend and im so happy")

# %% [markdown]
# the word "my" has the highest weight of 0.102 for the extracted topic.
# 
# A high weight for the word "my" suggests that it is a significant term within the topic identified by the LDA model. This means that the word "my" occurs frequently and carries substantial importance within the text when discussing the particular topic associated with that topic index.
# 
# In this case, it indicates that personal ownership or possession, likely related to the topic of loss and enduring emotional impact, plays a prominent role in the text. The word "my" may be indicating a personal connection or the speaker's individual perspective in relation to the topic being discussed.

# %% [markdown]
# ## 3.Sensory details 

# %%
import nltk

def analyze_sensory_details(text):
    sentences = nltk.sent_tokenize(text)

    sensory_keywords = {
        'sight': ['see', 'look', 'watch'],
        'sound': ['hear', 'listen', 'sound'],
        'smell': ['smell', 'scent', 'aroma'],
        'taste': ['taste', 'flavor'],
        'touch': ['feel', 'touch', 'texture']
    }

    sensory_details = []
    for sentence in sentences:
        lower_sentence = sentence.lower()
        for sense, keywords in sensory_keywords.items():
            for keyword in keywords:
                if keyword in lower_sentence:
                    sensory_details.append({'sense': sense, 'sentence': sentence})
                    break

    return sensory_details


# %%
def test_sensory_details(text):
    sensory_details = analyze_sensory_details(text)
    print("Sensory Details:")
    for detail in sensory_details:
        print(f"{detail['sense']}: {detail['sentence']}")

# %%
test_sensory_details(data.iloc[2]['Story'])

# %% [markdown]
# ## 4.Phenominal experience 

# %% [markdown]
# using emotional analysis 

# %%
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_emotional_tone(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

speech = "The loss of my father will forever leave an indelible mark on my heart. But im so happy that he died"

emotion_scores = analyze_emotional_tone(speech)
print("Emotion Scores:", emotion_scores)


# %% [markdown]
# ## Truthfulness

# %%



