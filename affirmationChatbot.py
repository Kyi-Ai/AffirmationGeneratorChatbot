import streamlit as st
import pandas as pd
import spacy
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

try:
    spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    spacy.load("en_core_web_sm")

# Load the model
nlp = spacy.load("en_core_web_sm")


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load datasets
data = pd.read_csv("/Users/minehantkyi/Desktop/Simbolo/ProjectCode/reduced_emotion_dataset.csv")  # Replace with your dataset path
affirmations = pd.read_csv("/Users/minehantkyi/Desktop/Simbolo/ProjectCode/affirmation new.csv")  # Replace with your affirmations path

# Preprocessing function
def preprocess_text_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Preprocess text data
data["Cleaned_Text"] = data["Text"].apply(preprocess_text_spacy)

# Feature extraction and model training
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["Cleaned_Text"]).toarray()
y = data["Mood"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Mood analysis function
def analyze_mood(user_input):
    cleaned_input = preprocess_text_spacy(user_input)
    input_vector = vectorizer.transform([cleaned_input]).toarray()
    mood_probabilities = model.predict_proba(input_vector)[0]
    mood_classes = model.classes_
    return {mood_classes[i]: mood_probabilities[i] for i in range(len(mood_classes))}

# Affirmation generation function
def generate_affirmation(mood_analysis):
    predicted_mood = max(mood_analysis, key=mood_analysis.get)
    filtered_affirmations = affirmations[affirmations["Mood"] == predicted_mood]["Affirmation"].tolist()
    return random.choice(filtered_affirmations) if filtered_affirmations else "Stay positive! You got this!"

# Streamlit UI
st.title("Affirmation Chatbot")
st.write("Describe your mood, and I'll generate a positive affirmation for you!")

# Input from user
user_input = st.text_area("How are you feeling today?")

if st.button("Get Affirmation"):
    if user_input:
        mood_analysis = analyze_mood(user_input)
        affirmation = generate_affirmation(mood_analysis)

        st.subheader("Mood Analysis:")
        for mood, prob in mood_analysis.items():
            st.write(f"{mood.capitalize()}: {prob*100:.2f}%")

        st.subheader("Affirmation:")
        st.write(affirmation)
    else:
        st.error("Please enter your mood description!")

st.markdown("---")
st.write("Built with ❤️ using Streamlit.")
