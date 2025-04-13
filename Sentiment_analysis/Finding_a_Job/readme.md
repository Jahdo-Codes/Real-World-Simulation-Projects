# Resume-to-Job Keyword Matcher

A smart, interactive NLP app that helps job seekers tailor their resumes to specific job descriptions. This tool analyzes keywords, identifies gaps, and calculates a match score — making your job applications more targeted and effective.

**Live Demo**: [Try it on Streamlit](https://sentiment-analysisfinding-a-jobskil-isnhbv.streamlit.app)

---

## Features

- Upload or paste resumes and job descriptions (PDF or text)
- Extract lemmatized, POS-filtered keywords (nouns, verbs, phrases)
- Generate word clouds from job and resume inputs
- Calculate a **Resume Match Score**
- Display matched vs missing keywords
- View top keywords from both sources

---

## NLP Pipeline

This app uses natural language processing (NLP) techniques via NLTK:

- Text cleaning (lowercase, punctuation removal)
- Tokenization & stopword removal
- POS tagging (nouns and verbs)
- Lemmatization
- Phrase extraction (bigrams and trigrams)

---

## Tech Stack

| Tool           | Purpose                        |
|----------------|--------------------------------|
| Python         | Core programming language      |
| Streamlit      | Interactive web interface      |
| NLTK           | NLP pipeline                   |
| WordCloud      | Keyword visualization          |
| PyMuPDF (fitz) | Extract text from PDFs         |
| Matplotlib     | Render word clouds             |

---

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
streamlit run "Sentiment analysis/Finding A Job/skills_matcher.py"
```

---

### Folder structure 

Sentiment analysis/
└── Finding A Job/
    ├── skills_matcher.py
    └── README.md

requirements.txt
README.md

---

## Author

**Jahdo Vanterpool**  
Poughkeepsie, NY  
[Email](jahdovpool@hotmail.com)  
[LinkedIn](https://www.linkedin.com/in/jahdo-vanterpool)  
[GitHub](https://github.com/Jahdo-Codes)

