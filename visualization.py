import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.utils import resample

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Clean text
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

# Load dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], axis=0).reset_index(drop=True)
df["combined"] = df["title"] + " " + df["text"]
df["clean_text"] = df["combined"].apply(clean_text)

# Balance dataset
df_fake = df[df["label"] == 0]
df_real = df[df["label"] == 1]
df_real_downsampled = resample(df_real, replace=True, n_samples=len(df_fake), random_state=42)
df_balanced = pd.concat([df_fake, df_real_downsampled]).sample(frac=1, random_state=42)

# Train-test split
X = df_balanced["clean_text"]
y = df_balanced["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Accuracy & RMSE
accuracy = accuracy_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ Accuracy:", round(accuracy * 100, 2), "%")
print("📉 RMSE:", round(rmse, 4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification report
report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
print("\n🔍 Classification Report:\n", report)

# Bar chart of metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = [round(accuracy * 100, 2), round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2)]
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(6, 4))
sns.barplot(x=labels, y=metrics, palette='Set2')
plt.ylim(0, 100)
plt.title("Model Performance Metrics")
for i, v in enumerate(metrics):
    plt.text(i, v + 1, f"{v}%", ha='center')
plt.tight_layout()
plt.show()
