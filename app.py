import os, re, string, joblib, pandas as pd, numpy as np, streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class DataPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'i', 'me', 'my', 'the', 'a', 'an', 'in', 'on', 'at', 'for'}
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None

    def clean_text(self, text):
        if not isinstance(text, str): return ''
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        words = text.split()
        return ' '.join([w for w in words if w not in self.stop_words])

from sklearn.metrics.pairwise import cosine_similarity

class InternshipRecommender:
    def __init__(self, data_path, comp_path):
        self.data = pd.read_pickle(data_path)
        comp = joblib.load(comp_path)
        self.vectorizer = comp['tfidf_vectorizer']
        self.feature_matrix = comp['feature_matrix']
        self.preprocessor = DataPreprocessor()

    def calculate_similarity(self, query):
        clean_q = self.preprocessor.clean_text(query)
        q_vec = self.vectorizer.transform([clean_q])
        return cosine_similarity(q_vec, self.feature_matrix).flatten()
