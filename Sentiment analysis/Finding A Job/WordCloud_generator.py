#The set up

import streamlit as st
import re
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

### Text processing functions

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]\s", " ", text)
    text = re.sub(r"\n", " ", text)
    words = word_tokenize(text)
    word = [word for word in words if word not in stop_words]
    return " ".join(word)

def get_top_keywords(text, num=10):
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return Counter(words).most_common(num)

def categorize_keywords(keywords, technical_skills, soft_skills):
    tech = [word for word in keywords if word in technical_skills]
    soft = [word for word in keywords if word in soft_skills]
    other = [word for word in keywords if word not in soft_skills and word not in technical_skills]
    return tech, soft, other

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

### The UI

st.title("Word Cloud Generator")

job_input = st.text_area("Paste the Job Description here:", height=200)
resume_input = st.text_area("Paste your Resume here:", height=200)


# Add sidebar
st.sidebar.header("Customize skill keywords")

#Checks if tech keywords have already been initialized in the session
if 'tech_keywords' not in st.session_state:
    st.session_state.tech_keywords = ['Team player, communication','proactive']

# Add new technical skills
tech_input = st.sidebar.text_input("Add to technical skills:")
if st.sidebar.button("Add to technical"):
    new_skill = tech_input.lower().strip()
    if new_skill and new_skill not in st.session_state.tech_keywords:
        st.session_state.tech_keywords.append(new_skill)

# Show current skills
st.sidebar.header("Current technical skills:")
for skill in st.session_state.tech_keywords:
    st.sidebar.write(f" {skill}")