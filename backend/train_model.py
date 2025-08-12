import os
import pandas as pd
import chardet
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset folder path
dataset_folder = r"A:\python\SPAM DETECTION\backend\dataset"

# Output model and vectorizer paths
model_path = r"A:\python\SPAM DETECTION\backend\spam_phishing_detector.pkl"
vectorizer_path = r"A:\python\SPAM DETECTION\backend\vectorizer.pkl"

# List to store all data
all_data = []

# Loop through all CSV files
for file in os.listdir(dataset_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(dataset_folder, file)

        # Detect encoding
        with open(file_path, "rb") as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)["encoding"]

        # Auto-fix CSV by rewriting it with quotes around text fields
        cleaned_file_path = file_path.replace(".csv", "_cleaned.csv")
        with open(file_path, "r", encoding=encoding, errors="ignore") as infile, \
                open(cleaned_file_path, "w", encoding="utf-8", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

            for row in reader:
                # Skip empty or malformed rows
                if len(row) < 2 or not row[0].strip() or not row[1].strip():
                    continue
                writer.writerow(row)

        # Load the cleaned CSV into DataFrame
        df = pd.read_csv(cleaned_file_path)

        # Ensure correct column names
        df.columns = [col.strip().lower() for col in df.columns]
        if "label" not in df.columns or "text" not in df.columns:
            raise ValueError(f"CSV file {file} must have columns 'label' and 'text'.")

        # Append to all_data list
        all_data.append(df)

# Combine all datasets into one DataFrame
if not all_data:
    raise ValueError("No valid CSV files found in the dataset folder.")

data = pd.concat(all_data, ignore_index=True)

# Remove duplicates & missing values
data.dropna(subset=["label", "text"], inplace=True)
data.drop_duplicates(inplace=True)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"✅ Model saved to {model_path}")
print(f"✅ Vectorizer saved to {vectorizer_path}")
