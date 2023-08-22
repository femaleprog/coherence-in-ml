#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


data = pd.read_json('/kaggle/input/stories/data_sensory_details.json')



#  Criteria for defining a personal event memory by Pillemer ( 1998 ) :  
# *  (a) present a specific event that took place at a particular time and place, rather than a summary event or extended series of events.
# * (b) contain a detailed account of the rememberer's own personal circumstances at the time of the event. 
# * (c) evoke sensory images or bodily sensations that contribute to the feeling of "re-experiencing" or "reliving" the event.
# * (d) link its details and images to a particular moment or moments of phenomenal experience. 
# * (e) be believed to be a truthful representation of what actually transpired.

# ## 1. Specificity

# In[4]:


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


# In[5]:


print(data.iloc[0]['Story'])


# In[6]:


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


# Testing the specifity_ne_spacy function :

# In[7]:


# Example usage
story_text = "Once upon a time, in a faraway land, there lived a brave knight named Sir Lancelot..."
is_specific_story = specificity_spacy(story_text)
print("Is the story specific?", is_specific_story)


# In[8]:


# Example usage
story_text = 'The annual science fair showcased impressive projects from young innovators. Students presented their research findings on various topics. A robot that can solve complex puzzles was a highlight of the event. Researchers discussed cutting-edge advancements in artificial intelligence. The fair concluded with an awards ceremony honoring the top projects.'
is_specific_story = specificity_spacy(story_text)
print("Is the story specific?", is_specific_story)


# In[ ]:


# Example usage
story_text = 'The annual science called Gitex fair showcased impressive projects from young innovators. Students presented their research findings on various topics. A robot that can solve complex puzzles was a highlight of the event. Researchers discussed cutting-edge advancements in artificial intelligence. The fair concluded with an awards ceremony honoring the top projects.'
is_specific_story = specificity_spacy(story_text)
print("Is the story specific?", is_specific_story)


# In[9]:


def evaluate_specificity_dataframe(data_frame):
    correct_predictions = 0
    total_stories = len(data_frame)

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['is_specific']
        
        predicted_label = specificity_spacy(story)
        
        # Convert boolean to string
        predicted_label_str = "yes" if predicted_label else "no"
        
        if predicted_label_str == ground_truth_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_stories

    return accuracy

accuracy = evaluate_specificity_dataframe(data)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:


def evaluate_precision(data_frame):
    true_positives = 0
    false_positives = 0

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['is_specific']

        predicted_label = specificity_spacy(story)
        
        if predicted_label and ground_truth_label == "yes":
            true_positives += 1
        elif predicted_label and ground_truth_label == "no":
            false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    return precision

precision = evaluate_precision(data)
print(f"Precision: {precision:.2f}")


# In[ ]:


def evaluate_recall(data_frame):
    true_positives = 0
    false_negatives = 0

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['is_specific']

        predicted_label = specificity_spacy(story)
        
        if predicted_label and ground_truth_label == "yes":
            true_positives += 1
        elif not predicted_label and ground_truth_label == "yes":
            false_negatives += 1
            print(row['story'])
    recall = true_positives / (true_positives + false_negatives)
    return recall

recall = evaluate_recall(data)
print(f"Recall: {recall:.2f}")


# Here's what recall reflects:
# 
# Completeness: Recall indicates how well your model is capturing all instances of the positive class (in your case, the specific stories). A higher recall means that your model is successfully identifying most of the specific stories present in the dataset.
# 
# Missed Positive Cases: A low recall value suggests that your model is missing a significant portion of the positive cases. This could mean that the model is not sensitive enough to detect the specific stories, leading to false negatives.
# 
# Trade-off with Precision: Recall is often in conflict with precision. A high recall could lead to more false positives (cases incorrectly classified as positive), as the model may be more inclusive in classifying instances as positive. Balancing recall and precision is important depending on your use case.

# # 2.Personal Context 

# In[ ]:


def personal_context(text):
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(sentence)["compound"] for sentence in sentences]

    # Filter emotional sentences based on sentiment threshold
    emotional_sentences = [sentence for i, sentence in enumerate(sentences) ]
    print(emotional_sentences)

    # Initialize the list to store teller's feelings
    teller_feelings = []
    
    # Pronoun and Verbal Analysis
    for sentence in emotional_sentences:
        words = nltk.word_tokenize(sentence.lower())
        person_pronoun = ["i", "me", "my", "mine"]
        verbal_indicators = ["felt", "was", "experienced", "sensed","loved","hated"]
        
        if any(pronoun in words for pronoun in person_pronoun ) and any(indicator in words for indicator in verbal_indicators):
            teller_feelings.append(sentence)
            #print(sentence)

    # Topic Modeling (only if there are teller's feelings)
    if teller_feelings:
        tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in teller_feelings]
        dictionary = corpora.Dictionary(tokenized_sentences)
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_sentences]
        lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary)

        # Extract most significant topics
        topics = [lda_model.get_document_topics(doc) for doc in corpus]
        most_significant_topics = [max(topic, key=lambda x: x[1]) for topic in topics]

        # Get actual topics
        actual_topics = [lda_model.print_topic(topic[0]) for topic in most_significant_topics]
    else:
        actual_topics = []

    # Return emotional sentences, sentiment scores, and actual topics
    return teller_feelings, sentiment_scores, actual_topics

# Call the function with your input text
result = personal_context("The mine annual science called Gitex fair showcased impressive projects from young innovators. Students presented their research findings on various topics. A robot that can solve complex puzzles was a highlight of the event. Researchers discussed cutting-edge advancements in artificial intelligence. The fair concluded with an awards ceremony honoring the top projects. i loved it")
print(result)


# In[ ]:


def has_personal_context(story, sentiment_threshold=0.5):
    # Run the personal_context function to get the results
    teller_feelings, sentiment_scores, actual_topics = personal_context(story)
    
    # Check if teller_stories is not empty and sentiment score is above threshold
    if teller_feelings and any(score > sentiment_threshold for score in sentiment_scores):
        return "yes"
    else:
        return "no"

# Call the has_personal_context function with your input story
result = has_personal_context("The annual science called Gitex fair showcased impressive projects from young innovators. Students presented their research findings on various topics. A robot that can solve complex puzzles was a highlight of the event. Researchers discussed cutting-edge advancements in artificial intelligence. The fair concluded with an awards ceremony honoring the top projects. i loved it")
print(result)


# In[ ]:


def accuracy_personal_context(data_frame):
    correct_predictions = 0
    total_stories = len(data_frame)

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['has_personal_context']
        
        predicted_label = has_personal_context(story)
        
        
        if predicted_label == ground_truth_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_stories

    return accuracy

accuracy = accuracy_personal_context(data)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:


def precision_personal_context(data_frame):
    
    true_positives = 0 
    false_positives = 0

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['has_personal_context']
        
        predicted_label = has_personal_context(story)
        
        
        if predicted_label and ground_truth_label == "yes":
            true_positives += 1
        elif predicted_label and ground_truth_label == "no":
            false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    return precision

precision = precision_personal_context(data)
print(f"Precision: {precision:.2f}")


# In[ ]:


def recall_personal_context(data_frame):
    true_positives = 0 
    false_negatives = 0 

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['has_personal_context']
        
        predicted_label = has_personal_context(story)
        
        
        if predicted_label and ground_truth_label == "yes":
            true_positives += 1
        elif not predicted_label and ground_truth_label == "yes":
            false_negatives += 1

    recall = true_positives / (true_positives + false_negatives)
    return recall

recall = recall_personal_context(data)
print(f"Recall: {recall:.2f}")


# ## 3.Sensory details 

# In[ ]:


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


# In[ ]:


has_sensory_details(data.iloc[2]['Story'])


# In[ ]:


def accuracy_sensory_details(data_frame):
    correct_predictions = 0
    total_stories = len(data_frame)

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['has_sensory_details']
        
        predicted_label = has_personal_context(story)
        
        
        if predicted_label == ground_truth_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_stories

    return accuracy

accuracy = accuracy_sensory_details(data)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:


def precision_sensory_details(data_frame):
    
    true_positives = 0 
    false_positives = 0

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['has_sensory_details']
        
        predicted_label = has_sensory_details(story)
        
        
        if predicted_label and ground_truth_label == "yes":
            true_positives += 1
        elif predicted_label and ground_truth_label == "no":
            false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    return precision

precision = precision_sensory_details(data)
print(f"Precision: {precision:.2f}")


# In[ ]:


def recall_sensory_details(data_frame):
    true_positives = 0 
    false_negatives = 0 

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['has_sensory_details']
        
        predicted_label = has_personal_context(story)
        
        
        if predicted_label and ground_truth_label == "yes":
            true_positives += 1
        elif not predicted_label and ground_truth_label == "yes":
            false_negatives += 1

    recall = true_positives / (true_positives + false_negatives)
    return recall

recall = recall_sensory_details(data)
print(f"Recall: {recall:.2f}")


# # Causal Coherence 

# In[ ]:




