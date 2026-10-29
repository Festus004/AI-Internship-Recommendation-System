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

    def recommend(self, query, min_stipend=0, max_duration=12, location=None, top_n=20):
        sim = self.calculate_similarity(query)
        df = self.data.copy()
        df['similarity_score'] = sim
        if min_stipend > 0:
            df = df[df['stipend_numeric'] >= min_stipend]
        if max_duration < 12:
            df = df[df['duration_numeric'] <= max_duration]
        if location and location != 'Any Location':
            df = df[df['location'].str.contains(location, case=False, na=False)]
        return df.sort_values(by='similarity_score', ascending=False).head(top_n)
