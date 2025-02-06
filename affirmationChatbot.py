# -*- coding: utf-8 -*-
"""Emotion Affirmation Generator - Streamlit App"""

import pandas as pd
import spacy
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Error: spaCy model 'en_core_web_sm' is not installed. Run: python -m spacy download en_core_web_sm")
    st.stop()

# Function to generate affirmations
def generate_affirmation(emotion):
    affirmations = {
        "happy": ["You are a source of joy!", "Your happiness is contagious!"],
        "sad": ["It's okay to feel sad. You are not alone.", "Better days are coming!"],
        "angry": ["Take a deep breath. You are in control.", "Let go of anger, embrace peace."],
    }
    return random.choice(affirmations.get(emotion, ["You are strong and capable!"]))

# Streamlit UI
st.title("Positive Affirmation Generator")
st.write("Select your emotion to receive an affirmation.")

# Emotion selection
emotion = st.selectbox("How are you feeling today?", ["happy", "sad", "angry"])

if st.button("Generate Affirmation"):
    affirmation = generate_affirmation(emotion)
    st.success(affirmation)
