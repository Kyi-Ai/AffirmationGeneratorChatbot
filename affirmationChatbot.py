# -*- coding: utf-8 -*-
"""Emotion Affirmation Generator"""

import pandas as pd
import spacy
import subprocess
import random
import os
from openpyxl import Workbook, load_workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import schedule
import time
import smtplib
from email.mime.text import MIMEText
import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load spaCy model
spacy_model = "en_core_web_sm"
try:
    nlp = spacy.load(spacy_model)
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
    nlp = spacy.load(spacy_model)

# Step 2: Load Datasets
data = pd.read_csv('reduced_emotion_dataset.csv')  # Mood Dataset
affirmations = pd.read_csv('affirmation_new.csv')  # Affirmation Dataset

# Step 3: Preprocess Text Data with spaCy
def preprocess_text_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return ' '.join(tokens)

data['Cleaned_Text'] = data['Text'].apply(preprocess_text_spacy)

# Step 4: Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Cleaned_Text']).toarray()
y = data['Mood']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
st.write("Classification Report:\n", classification_report(y_test, y_pred))
st.write("Accuracy Score:", (accuracy_score(y_test, y_pred))*100, "%")

# Step 8: Analyze the Mood from User Input
def analyze_mood(user_input):
    cleaned_input = preprocess_text_spacy(user_input)
    input_vector = vectorizer.transform([cleaned_input]).toarray()
    mood_probabilities = model.predict_proba(input_vector)[0]
    mood_classes = model.classes_
    mood_analysis = {mood_classes[i]: mood_probabilities[i] for i in range(len(mood_classes))}
    return mood_analysis

# Step 9: Generate Affirmations
def generate_affirmation(mood_analysis):
    predicted_mood = max(mood_analysis, key=mood_analysis.get)
    filtered_affirmations = affirmations[affirmations['Mood'] == predicted_mood]['Affirmation'].tolist()
    return random.choice(filtered_affirmations)

# Step 10: Save User Email and Affirmation
def save_email_and_affirmation(email, affirmation):
    file_path = 'user_affirmations.xlsx'

    if not os.path.exists(file_path):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Email", "Affirmation"])
        workbook.save(file_path)

    workbook = load_workbook(file_path)
    sheet = workbook.active
    sheet.append([email, affirmation])
    workbook.save(file_path)
    st.write(f"Your affirmation has been saved and will be sent to {email} tomorrow morning.")

# Step 11: Send Affirmation Email
def send_email(email, affirmation):
    sender_email = "posivibescorner@gmail.com"
    sender_password = "PosivibesCorner1000"
    subject = "Your Morning Affirmation \ud83c\udf1e"
    body = f"Good morning!\n\nHere’s an affirmation for you today:\n\n{affirmation}\n\nHave a wonderful day!"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
            st.write(f"Affirmation sent to {email}")
    except Exception as e:
        st.write("Error sending email:", e)

# Step 12: Schedule Email Sending
def send_scheduled_emails():
    file_path = 'user_affirmations.xlsx'
    if not os.path.exists(file_path):
        st.write("No emails to send.")
        return

    workbook = load_workbook(file_path)
    sheet = workbook.active

    for row in sheet.iter_rows(min_row=2, values_only=True):
        email, affirmation = row
        send_email(email, affirmation)

    # Clear the file after sending all emails
    sheet.delete_rows(2, sheet.max_row)
    workbook.save(file_path)

# Streamlit App
st.title("Emotion Affirmation Generator")

user_input = st.text_input("How are you feeling today? Describe your mood:")
email = st.text_input("Please enter your email address:")

if st.button("Generate Affirmation"):
    if user_input and email:
        predicted_mood = analyze_mood(user_input)
        affirmation = generate_affirmation(predicted_mood)
        st.write(f"Your affirmation: {affirmation}")
        save_email_and_affirmation(email, affirmation)
    else:
        st.write("Please fill in both fields.")

# Schedule the email sending at 7 AM every day
schedule.every().day.at("07:00").do(send_scheduled_emails)

# Run the scheduler (in a long-running process)
while True:
    schedule.run_pending()
    time.sleep(1)
