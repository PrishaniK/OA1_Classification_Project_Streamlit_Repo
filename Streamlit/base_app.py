"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

import streamlit as st
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pandas as pd
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define path to the directory containing all pickled files
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

# Paths to pickled models and vectorizer
logistic_regression_path = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
svm_path = os.path.join(MODEL_DIR, 'svm_model.pkl')
naive_bayes_path = os.path.join(MODEL_DIR, 'naive_bayes_model.pkl')
tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Define text cleaning function
def clean_text(text, is_url=False):
    if isinstance(text, str):
        # Handle the URL in the text
        if is_url:
            text = re.sub(r'https?://[^/]+/', '', text)
            text = re.sub(r'[^a-zA-Z0-9]', ' ', text) 
        else:
            # Lowercase the text
            text = text.lower()
            
            # Remove punctuation using string
            text = text.translate(str.maketrans('', '', string.punctuation))

            # Remove numerical values in the text
            text = re.sub(r'\d+', '', text)
        
            # Tokenize the text
            tokens = word_tokenize(text)
        
            # Remove stop words
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        
            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
            # Remove special characters and extra whitespeaces
            tokens = [re.sub(r'\W+', '', word) for word in tokens]
            
            # Filter out empty strings and single letter words
            tokens = [word for word in tokens if word and len(word) > 1]
            
            # Ensure that words are written as individual words
            distinct_tokens = set(tokens)
        
            # Join the tokens back together
            text = ' '.join(distinct_tokens)
        
        return text 
    else:  
        return text
    
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = joblib.load(file)
    return model



def main():
    """News classifier"""

    # Load vectorizer
    with open(tfidf_vectorizer_path, 'rb') as file:
        tfidf_vectorizer = joblib.load(file)
    
    # Define available models
    models = {
        "Logistic Regression": logistic_regression_path,
        "Support Vector Machine": svm_path,
        "Naive Bayes": naive_bayes_path
    }
	
	# Define path to the training dataset
    train_data_path = "train.csv"

    # Load your raw data
    df = pd.read_csv(train_data_path)
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Feast")
    st.subheader("Analysing news articles")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]

    # Ensure each widget has a unique key to avoid DuplicateWidgetID error
    selection = st.sidebar.selectbox("Choose Option", options)

    # Model selection dropdown
    selected_model = st.sidebar.selectbox("Choose Model", list(models.keys()))

    # Load selected model
    model_path = models[selected_model]
    model = load_model(model_path)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("""
            "Explore the world of news with our innovative app! From Business to Technology, Sports to Entertainment, and beyond, discover insights from a diverse collection of articles. Our app uses advanced machine learning to predict categories for new articles, ensuring you stay informed effortlessly. Experience seamless interaction and practical utility, making knowledge accessible anytime, anywhere. Dive into the future of news analysis with us!"
            This application demonstrates a news classifier using different machine learning models. (Currently only trained using Business, Entertainment, Technology, Education and Sport categories.)
            
            ### How to Use:
            - **Prediction**: Enter text in the text box and select a model from the dropdown to classify it.
            - **Models Available**: Choose from Logistic Regression, Support Vector Machine, or Naive Bayes.
            
            ### About the Models:
            - **Logistic Regression**: A linear model suitable for binary classification tasks.
            - **Support Vector Machine (SVM)**: Effective for both linear and non-linear classification.
            - **Naive Bayes**: Assumes independence between features, often used for text classification.
            
            For more information on Streamlit, refer to the [Streamlit Documentation](https://docs.streamlit.io/en/latest/).
        """)

        
    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with {} Model".format(selected_model))
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tfidf_vectorizer.transform([news_text]).toarray()
            # Make predictions using selected model
            prediction = model.predict(vect_text)
            
            # When model has successfully run, will print prediction
            st.success("Text Categorized as: {}".format(prediction[0]))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
