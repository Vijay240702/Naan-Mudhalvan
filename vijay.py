import streamlit as st
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Text Cleaning Function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+|www.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub(r"\n", "", text)
        text = re.sub(r"\w*\d\w*", "", text)
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text
    return ""

# Load Dataset
fake = pd.read_csv(r"V:\NAAN\Fake.csv")
real = pd.read_csv(r"V:\NAAN\True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], axis=0).reset_index(drop=True)
df["combined"] = df["title"] + " " + df["text"]
df["clean_text"] = df["combined"].apply(clean_text)

# Balance Dataset
df_fake = df[df["label"] == 0]
df_real = df[df["label"] == 1]
df_real_downsampled = resample(df_real, replace=True, n_samples=len(df_fake), random_state=42)
df_balanced = pd.concat([df_fake, df_real_downsampled]).sample(frac=1, random_state=42)

# Train/Test Split
X = df_balanced["clean_text"]
y = df_balanced["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.7, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train Model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Streamlit App
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news article below and check if it's **Real** or **Fake**:")

user_input = st.text_area("Enter News Text:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        input_tfidf = vectorizer.transform([cleaned])
        prediction = model.predict(input_tfidf)[0]
        result = "🟢 REAL News" if prediction == 1 else "🔴 FAKE News"
        st.success(f"Prediction: {result}")

