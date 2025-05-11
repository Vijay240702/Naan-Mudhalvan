import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import resample

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Text cleaning function
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

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Label them
fake['label'] = 0
real['label'] = 1

# Combine and shuffle
df = pd.concat([fake, real], axis=0).reset_index(drop=True)
df['combined'] = df['title'] + " " + df['text']
df['clean_text'] = df['combined'].apply(clean_text)

# Balance dataset
df_fake = df[df['label'] == 0]
df_real = df[df['label'] == 1]
df_real_downsampled = resample(df_real, replace=True, n_samples=len(df_fake), random_state=42)
df_balanced = pd.concat([df_fake, df_real_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split data
X = df_balanced['clean_text']
y = df_balanced['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.7, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy and RMSE
accuracy = accuracy_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ Accuracy:", round(accuracy * 100, 2), "%")
print("📉 RMSE:", round(rmse, 4))
