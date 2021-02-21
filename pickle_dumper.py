## importing
import pandas as pd
import numpy as np
import urllib.request
import json
import string
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import nltk
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from google_trans_new import google_translator

## reading data
df = pd.read_json('https://talentbait-assets.s3.eu-central-1.amazonaws.com/tech_soft_none.json')
# df = pd.read_json('dataset.json')
df = pd.json_normalize(df['data'])
df.head()

## defining names
text = df['text']
label = df['label']

## helpers for data cleaning
punctuation = string.punctuation
stop_words = list(STOP_WORDS)
nlp = spacy.load('de_core_news_sm')

## function that cleans input text
def cleaning_function(input_text):
    text = nlp(input_text)
    tokens = []
    for token in text:
        temp = token.lemma_.lower()
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and token not in punctuation:
            cleaned_tokens.append(token)
    return cleaned_tokens

X = text
y = label

## SVC using tfidf 
## ONLY TRAINING MODELS
tfidf = TfidfVectorizer(tokenizer = cleaning_function)
classifier = LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state=50)
# print(X_train.shape, X_test.shape)
SVC_clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
SVC_clf.fit(X_train, y_train)

def predict_class(user_input_from_function):
    translator = google_translator()
    if(translator.detect(user_input_from_function)[0]!='de'):
        user_input_from_function = translator.translate(user_input_from_function, lang_tgt='de')
    return (SVC_clf.predict([user_input_from_function])[0])

## pickle for command line application
import pickle
with open('trained_model', 'wb') as f:
    pickle.dump(SVC_clf, f)