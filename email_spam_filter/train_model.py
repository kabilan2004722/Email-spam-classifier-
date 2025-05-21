import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the CSV file with commas (default)
df = pd.read_csv('spam.csv')

# Rename columns for clarity
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Drop rows where message is empty or missing
df = df.dropna(subset=['message'])
df = df[df['message'].str.strip().astype(bool)]

# Convert message column to string type
df['message'] = df['message'].astype(str)

# Convert label 'ham' to 0, 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize the message text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

# Labels
y = df['label']

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Save the model and vectorizer to files
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training completed and model saved.")
