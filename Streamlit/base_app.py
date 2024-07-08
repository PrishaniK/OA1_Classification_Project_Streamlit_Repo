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
# # Streamlit dependencies
import streamlit as st
import joblib
import os
import pandas as pd
from PIL import Image
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define path to the directory containing all pickled files
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

# Paths to pickled models and vectorizer
logistic_regression_path = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
svm_path = os.path.join(MODEL_DIR, 'svm_model.pkl')
naive_bayes_path = os.path.join(MODEL_DIR, 'naive_bayes_model.pkl')
tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Load vectorizer
with open(tfidf_vectorizer_path, 'rb') as file:
    tfidf_vectorizer = joblib.load(file)

# Load models
models = {
    "Logistic Regression": joblib.load(logistic_regression_path),
    "Support Vector Machine": joblib.load(svm_path),
    "Naive Bayes": joblib.load(naive_bayes_path)
}

# Load data
train_data_path = os.path.join(MODEL_DIR, 'train.csv')
df = pd.read_csv(train_data_path)

def main():
    st.title("News Feast")
    st.subheader("Analyzing news articles")

    # Displaying the picture under the title and subheader
    st.image(Image.open(os.path.join(MODEL_DIR, 'vintage-newspaper.jpg')), use_column_width=True)

    options = ["Prediction", "Information", "Feedback", "About Us"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Information":
        st.info("General Information")
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

    elif selection == "Prediction":
        st.info("Prediction")
        selected_model = st.selectbox("Choose Model", list(models.keys()))

        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tfidf_vectorizer.transform([news_text]).toarray()

            # Make predictions using selected model
            model = models[selected_model]
            prediction = model.predict(vect_text)

            # When model has successfully run, will print prediction
            st.success("Text Categorized as: {}".format(prediction[0]))

    elif selection == "Feedback":
        st.info("Feedback")
        st.markdown("""
            Your feedback is valuable to us! Please provide your comments and suggestions in the text area to help us improve this application.
            """
        )
        feedback_text = st.text_area("Your Feedback", "")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

    elif selection == "About Us":
        st.info("About Us")
        st.markdown("""

            For any inquiries or further information, please contact any one from our time: 

            ---
            **Our Team:**
            - Koketso Bambo	    moraka1952@gmail.com
            - Thobile Mvuni	    thoyomvuni@gmail.com
            - Prishani Kisten	    prishanik135@gmail.com
            - Chuma Gqola	        chuma.gqolacg@gmail.com
            - Kgolo Motshegoa	    motshegoakgolo@gmail.com
            - Lulama Mulaudzi	    lulamamulaudzi@gmail.com

            **Vision:**
            "To revolutionize information access by seamlessly categorizing news articles, empowering users with accurate and insightful content classification."
        """)

if __name__ == '__main__':
    main()