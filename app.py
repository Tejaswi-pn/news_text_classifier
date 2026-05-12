import streamlit as st
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
dataset = "https://raw.githubusercontent.com/ronaldleonardo/Data_Transformer_Datasets/refs/heads/main/bbc_text_cls.csv"
df = pd.read_csv(dataset)

# Features and Targetgit remote add origin https://github.com/Tejaswi-pn/news_text_classifier.git
X = df["text"]
y = df["labels"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Text Vectorization
cv = CountVectorizer(stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# Model Accuracy
y_pred = model.predict(X_test_cv)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("BBC News Text Classifier")
st.write("Predict the category of news text using Naive Bayes")
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# Text input
user_input = st.text_area("Enter News Text")

# Prediction button
if st.button("Predict Category"):

    if user_input.strip() != "":
        # Transform input text
        input_data = cv.transform([user_input])

        # Predict
        prediction = model.predict(input_data)

        st.success(f"Predicted Category: {prediction[0]}")

    else:
        st.warning("Please enter some text")
