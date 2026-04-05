import streamlit as st
import numpy as np
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # lowercase
    text = text.lower()
    
    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # tokenize
    words = text.split()
    
    # remove stopwords + lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Load data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('archive/TMDB_IMDB_Movies_Dataset.csv')

    # Due to the high amount of data, we would like to use only the top 20000 based on the imdb Weighted Average Ratings
    m = df['vote_count'].quantile(0.9)
    C = df['vote_average'].mean()

    # Defining Formula for calculating the score of each movie
    def weight_average(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m))*R + (m/(v+m))*C

    # Filtering Dataframe to get movies with vote count >= m
    q_movies = df.copy().loc[df['vote_count'] >= m]

    # Adding new feature 'score' according to IMDB formula to rank these movies rating-wise
    q_movies['score'] = q_movies.apply(weight_average, axis=1)

    # Sorting According to 'Score' attribute
    q_movies = q_movies.sort_values('score', ascending=False)

    # Keep top 20,000
    df_model = q_movies.sort_values('score', ascending=False).head(20000).copy()

    # Remove duplicate titles
    df_model = df_model.drop_duplicates(subset='title').reset_index(drop=True)

    # Heavy cleaning for text fields
    for feature in ['overview']:
        df_model[feature] = df_model[feature].fillna('').apply(clean_text)

    # Light cleaning for categorical text
    for feature in ['genres', 'keywords']:
        df_model[feature] = df_model[feature].fillna('').str.lower()

    # Special handling for names
    for feature in ['cast', 'directors']:
        df_model[feature] = df_model[feature].fillna('').str.lower()
        df_model[feature] = df_model[feature].str.replace(' ', '', regex=False)

    df_model['soup'] = (
        df_model['overview'] + ' ' +
        df_model['genres'] + ' ' +
        df_model['keywords'] + ' ' +
        df_model['cast'] + ' ' +
        df_model['directors']
    )

    return df_model

df_model = load_and_process_data()

# 1. Load a pretrained Sentence Transformer model
@st.cache_data
def compute_embeddings_and_similarities(df_model):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(
        df_model['soup'].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)

    return similarities

similarities = compute_embeddings_and_similarities(df_model)

indices = pd.Series(df_model.index, index=df_model['title']).drop_duplicates()

def get_recommendations(title, similarities, df_model, top_n=10):
    idx = indices[title]
    sim_scores = list(enumerate(similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    scores = [float(i[1]) for i in sim_scores]

    result = df_model.iloc[movie_indices][['title']].copy()
    result['similarity_score'] = scores
    return result

# Streamlit App
st.title("Movie Recommendation System")

movie_list = df_model['title'].tolist()
selected_movie = st.selectbox("Choose a movie to get recommendations:", movie_list)

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie, similarities, df_model)
    st.write("Recommended Movies:")
    st.dataframe(recommendations)