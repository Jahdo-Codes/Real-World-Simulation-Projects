import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import json #Files saved from google takeout were in json format
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words(['english','spanish']))
lemmatizer = WordNetLemmatizer()

# Custom function (already defined in your pipeline)
def find_comments(keywords, df, column='clean_text', n=50):
    if isinstance(keywords, str):
        keywords = [keywords]
    pattern = '|'.join(keywords)
    matches = df[df[column].str.contains(pattern, case=False, na=False)]
    if matches.empty:
        return f"No matches found for: {', '.join(keywords)}"
    return matches[[column]].sample(min(n, len(matches)))

# Load Data
st.title("📊 Local Business Review Sentiment Dashboard")
st.markdown("Analyze your Google reviews to uncover what customers love or dislike.")

uploaded_file = st.file_uploader("Upload Cleaned Reviews CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Rating Distribution")
    st.bar_chart(df['Stars'].value_counts().sort_index())

    st.subheader("3. Top Keywords")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Positive Keywords**")
        pos_keywords = df[df['Stars'] >= 4]['clean_text'].str.cat(sep=' ')
        wordcloud = WordCloud(stopwords='english', background_color='white').generate(pos_keywords)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.markdown("**Negative Keywords**")
        neg_keywords = df[df['Stars'] <= 2]['clean_text'].str.cat(sep=' ')
        wordcloud = WordCloud(stopwords='english', background_color='white').generate(neg_keywords)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    st.subheader("4. Search Comments by Keyword")
    keyword = st.text_input("Enter word or phrase to search for:")
    if keyword:
        results = find_comments(keyword, df)
        st.write(results)

    st.subheader("5. Download Processed File")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="processed_reviews.csv")
else:
    st.warning("👆 Upload a CSV file to get started.")

