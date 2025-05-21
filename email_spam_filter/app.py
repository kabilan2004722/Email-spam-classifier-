# app.py
from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
import os
import sys

app = Flask(__name__)

# Download stopwords once before using them
nltk.download('stopwords')

try:
    # Load model and vectorizer
    model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model or vectorizer file not found. Please run train_model.py first.", file=sys.stderr)
    sys.exit(1)

# Prepare stopwords set
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean input text by lowercasing, removing punctuation, and stopwords."""
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)
