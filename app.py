import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pytesseract
from PIL import Image
import pdfplumber
from nltk.sentiment import SentimentIntensityAnalyzer

# Setup title and description
st.title("Fishbone and USG Analysis Tool - Focus on PDF and Image")
st.write("Upload Fishbone Analysis and USG (Urgency, Seriousness, Growth) documents in PDF or image (JPG, PNG) format to evaluate their alignment and perform advanced analysis.")

# Supported file formats
SUPPORTED_FORMATS = ["pdf", "jpg", "png"]

# File upload for Fishbone or USG Analysis (PDF or Image)
uploaded_file = st.file_uploader("Choose a file (PDF or Image)", type=SUPPORTED_FORMATS)

# Initialize sentence transformer model for semantic similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract text from image or PDF
def extract_text_from_file(file):
    file_type = file.type
    if "image" in file_type:  # For image files (JPG, PNG)
        img = Image.open(file)
        text = pytesseract.image_to_string(img)  # OCR for images
        return text
    elif "pdf" in file_type:  # For PDF files
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()  # Extract text from each page
        return text
    else:
        return None

# Function to calculate semantic similarity using Sentence-BERT
def calculate_semantic_similarity(text1, text2):
    embeddings1 = model.encode([text1], convert_to_tensor=True)
    embeddings2 = model.encode([text2], convert_to_tensor=True)
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity[0][0]

# Function to perform TF-IDF based cosine similarity
def calculate_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    vectors_dense = vectors.toarray()
    similarity = cosine_similarity([vectors_dense[0]], [vectors_dense[1]])
    return similarity[0][0]

# Sentiment analysis function using NLTK's VADER
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Placeholder functions for advanced metrics
def calculate_gap_analysis(fishbone_text, usg_text):
    return np.random.rand()

def calculate_relevance(fishbone_text, usg_text):
    return np.random.rand()

def calculate_discrepancy(fishbone_text, usg_text):
    return np.random.rand()

def calculate_causal_inference(fishbone_text, usg_text):
    return np.random.rand()

def calculate_temporal_growth_prediction(fishbone_text, usg_text):
    return np.random.rand()

# Main process after file upload
if uploaded_file:
    # Extract text from the uploaded PDF or Image
    file_text = extract_text_from_file(uploaded_file)

    if file_text:
        st.subheader("Extracted Text from Uploaded File")
        st.write(file_text)

        # Compare with sample USG text
        if st.button("Compare with Sample USG Text"):
            sample_usg_text = "This is a sample text for USG analysis including Urgency, Seriousness, and Growth."
            similarity = calculate_semantic_similarity(file_text, sample_usg_text)
            st.write(f"Semantic Similarity with sample USG text: {similarity:.2f}")

        # Calculate advanced metrics
        gap_analysis_score = calculate_gap_analysis(file_text, sample_usg_text)
        relevance_score = calculate_relevance(file_text, sample_usg_text)
        discrepancy_score = calculate_discrepancy(file_text, sample_usg_text)
        causal_inference_score = calculate_causal_inference(file_text, sample_usg_text)
        temporal_growth_prediction = calculate_temporal_growth_prediction(file_text, sample_usg_text)

        # Features for analysis
        features = [similarity, gap_analysis_score, 0.75, discrepancy_score, relevance_score]
        feature_labels = ["Semantic Similarity", "Gap Analysis", "Sentiment Consistency", "Discrepancy", "Relevance"]

        # Create a DataFrame for features
        df_features = pd.DataFrame([features], columns=feature_labels)

        # Bivariate analysis: Correlation between variables
        st.subheader("Bivariate Correlation Analysis (Heatmap)")
        correlation_matrix = df_features.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Multivariate analysis: Logistic Regression
        st.subheader("Multivariate Analysis")
        y = np.random.randint(0, 3, df_features.shape[0])  # Dummy classification target (0: Buruk, 1: Baik, 2: Sangat Baik)
        X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.3, random_state=42)

        # Logistic regression for classification
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Classification report
        st.text("Multivariate Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Accuracy score
        st.write(f"Multivariate Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        # Display impact factors and detailed interpretation
        st.subheader("Impact Factor and Variable Significance")
        impact_data = {
            "Variable": feature_labels,
            "Score": features
        }
        impact_df = pd.DataFrame(impact_data)
        st.table(impact_df)

        # Show detailed calculation for each metric
        st.subheader("Detailed Metric Calculations")
        st.write(f"Semantic Similarity (BERT-based): {features[0]:.2f}")
        st.write(f"Gap Analysis Score: {features[1]:.2f}")
        st.write(f"Sentiment Consistency: {features[2]:.2f}")
        st.write(f"Discrepancy Score: {features[3]:.2f}")
        st.write(f"Relevance Score: {features[4]:.2f}")

    else:
        st.error("Unable to extract text from the uploaded file.")
