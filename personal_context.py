
#imports 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('vader_lexicon')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import spacy
from gensim import corpora, models

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