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

@st.cache_resource(show_spinner=False)
def load_recommender():
    data_pkl = 'internship_recommender_data_20251025_072624.pkl'
    comp_pkl = 'internship_recommender_components_20251025_072624.pkl'
    return InternshipRecommender(data_pkl, comp_pkl)

st.set_page_config(page_title='InternMatch AI | Intelligent Opportunity Platform', page_icon='🎓', layout='wide')

# Domain Presets
DOMAIN_PRESETS = {
    'All Opportunities': '',
    '🤖 AI & Machine Learning': 'Machine Learning, Deep Learning, Python, Data Science',
    '🌐 Web Development': 'React, JavaScript, Node.js, Python, Full Stack',
    '📊 Data Science': 'Data Science, Python, SQL, Statistics, Pandas',
    '☁️ Cloud & DevOps': 'DevOps, Docker, Kubernetes, Cloud Computing, Linux, AWS',
    '📱 Mobile Apps': 'Flutter, Android, React Native, Mobile Development',
    '📡 IoT & Embedded (CCE)': 'IoT, Embedded Systems, Arduino, Raspberry Pi, Communication',
    '🎨 UI/UX Design': 'UI/UX, Graphic Design, Figma, Adobe XD'
}

if 'has_searched' not in st.session_state:
    st.session_state.has_searched = False

def render_internship_card(row, idx):
    score = int(round(row.get('similarity_score', 0) * 100))
    st.markdown(f'### {row["internship_title"]} at {row["company_name"]} ({score}% Match)')
