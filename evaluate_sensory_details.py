import sys
sys.path.append('C:/Users/tichi/career/internships/internship2023/first_assignment/code/coherence-in-ml')
from sentiment_analysis.sensory_details import has_sensory_details
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
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
data = pd.read_json('./data/stories.json')

def accuracy_sensory_details(data_frame):
    correct_predictions = 0
    total_stories = len(data_frame)

    for index, row in data_frame.iterrows():
        story = row['story']
        ground_truth_label = row['has_sensory_details']
        
        predicted_label = has_sensory_details(story)
        
        
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
        
        predicted_label = has_sensory_details(story)
        
        
        if predicted_label and ground_truth_label == "yes":
            true_positives += 1
        elif not predicted_label and ground_truth_label == "yes":
            false_negatives += 1

    recall = true_positives / (true_positives + false_negatives)
    return recall

recall = recall_sensory_details(data)
print(f"Recall: {recall:.2f}")