#The set up

import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.util import ngrams

import fitz

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------- Utility Functions -----------
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\n", " ", text)
    return text

def tokenize_and_filter(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tagged = pos_tag(tokens)
    filtered = [lemmatizer.lemmatize(word) for word, tag in tagged if tag.startswith('N') or tag.startswith('V')]

    # Extract bigrams and trigrams (multi-word phrases)
    bigrams = [" ".join(gram) for gram in ngrams(filtered, 2)]
    trigrams = [" ".join(gram) for gram in ngrams(filtered, 3)]

    return filtered + bigrams + trigrams

def get_top_keywords(tokens, num=20):
    return Counter(tokens).most_common(num)

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.subheader(title)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

### The UI

st.title("Word matcher")

st.title("Resume-to-Job Keyword Matcher")

st.subheader("Select Input Method for Each Document")
job_input_type = st.radio("Job Description Input:", ("Paste Text", "Upload PDF"), key="job_input_type")
resume_input_type = st.radio("Resume Input:", ("Paste Text", "Upload PDF"), key="resume_input_type")

job_input = ""
resume_input = ""

if job_input_type == "Paste Text":
    job_input = st.text_area("Paste Job Description Here:", height=200, key="job_input")
else:
    job_pdf = st.file_uploader("Upload Job Description PDF", type=["pdf"], key="job_pdf")
    if job_pdf:
        job_input = extract_text_from_pdf(job_pdf)

if resume_input_type == "Paste Text":
    resume_input = st.text_area("Paste Your Resume Here:", height=200, key="resume_input")
else:
    resume_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume_pdf")
    if resume_pdf:
        resume_input = extract_text_from_pdf(resume_pdf)

### Analyze

if st.button("Analyze"):
    if job_input and resume_input:
        job_clean = clean_text(job_input)
        resume_clean = clean_text(resume_input)

        job_tokens = tokenize_and_filter(job_clean)
        resume_tokens = tokenize_and_filter(resume_clean)

        job_keywords = set(job_tokens)
        resume_keywords = set(resume_tokens)

        matched_keywords = job_keywords & resume_keywords
        missing_keywords = job_keywords - resume_keywords
### MAtch score
        match_score = round((len(matched_keywords) / len(job_keywords)) * 100, 2) if job_keywords else 0
        st.metric(label="✅ Resume Match Score", value=f"{match_score}%")

### Word cloud
        st.subheader("Word Clouds")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Job Description:")
            generate_wordcloud(job_clean,title="Job Description")
        with col2:
            st.markdown("Your Resume:")
            generate_wordcloud(resume_clean,title="Your Resume")

        ### Match results

        st.subheader(" Resume to job keyword match")
        st.markdown(f" Matched keywords ({len(job_keywords)}):")
        if matched_keywords:
            st.success(", ".join(sorted(matched_keywords)))
        else:
            st.info("No matching keywords found")

        st.markdown(f" Missing keywords ({len(missing_keywords)}):")
        if missing_keywords:
            st.warning(", ".join(sorted(missing_keywords)))
        else:
            st.success("You included all job-related keywords in your resume")

### Top keywords
        st.subheader("Top keywords")
        job_top = get_top_keywords(job_tokens)
        resume_top = get_top_keywords(resume_tokens)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Top Job Description:")
            for word, freq in job_top:
                st.write(f" {word}: {freq}")
        with col2:
            st.markdown("Top Resume Description:")
            for word, freq in resume_top:
                st.write(f" {word}: {freq}")


    else:
        st.warning("Please fill in botht the job description and resume text box")






