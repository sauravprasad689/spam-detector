import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load dataset (no header, tab-separated)
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])

# Step 2: Convert labels ('ham' → 0, 'spam' → 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Vectorize with TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Step 5: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model saved as spam_model.pkl")
print("✅ Vectorizer saved as vectorizer.pkl")
