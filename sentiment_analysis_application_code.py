import keras
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np
import re
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
from bs4 import BeautifulSoup

import sys
import os

# Set the default encoding to UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Define the header text with custom color using HTML
header_html1 = """
    <h2 style="color: darkblue;">Welcome to Sentiment Analysis App</h2>
"""

# Render the HTML in Streamlit
st.markdown(header_html1, unsafe_allow_html=True)

image = Image.open("C:/Users/Hp/Desktop/Codes/Sentiment analysis/1691232378248.jpeg")
st.image(image, caption='How are you feeling ? Lets check it out')

# Loading model
# model = keras.layers.TFSMLayer(filepath="C:/Users/Hp/Downloads/text_sentiment_analysis.h5")
model = load_model('C:/Users/Hp/Downloads/text_sentiment_analysis.h5')

def preprocess_text(text):
    # Convert the text to UTF-8 encoding
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Convert to lowercase
    text = text.lower()

    # Remove emojis
    text = emoji.demojize(text)

    # Remove special characters and number
    text = re.sub(r'[@#%&*^$£!()-_+={}\[\]:;<>,.?\/\\\'"`~]', '', text)

    # Remove special numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    filtered_words = ' '.join(filtered_words)

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmetize_words = [lemmatizer.lemmatize(word) for word in words]

    # Rejoin words to form the final processed text
    processed_text = ' '.join(lemmetize_words)

    return processed_text

# Making prediction on a processed_text
def predict_sentiment(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    
    # Load or define your tokenizer
    tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(cleaned_text)

    # Convert the text to sequences
    input_sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad the sequence to ensure it has the same length as the input the model expects
    max_len = 100  # model's expected input length
    padded_sequence = pad_sequences(input_sequence, maxlen=max_len, padding = 'post')
    
    # Predict the sentiment
    prediction = model.predict(padded_sequence)
    return prediction

# Streamlit app
header_html3 = """
    <h3 style="color: darkblue;">Enter a text to analyze its sentiment 😉</h3>
"""

# Render the HTML in Streamlit
st.markdown(header_html3, unsafe_allow_html=True)

user_input = st.text_area("Enter your text here:")
class_name = ['Fear','Surprise','Sadness','Joy','Love','Anger']
if st.button("Analyze"):
    if user_input:
        result = predict_sentiment(user_input)
        string="You are having sentiment of "+class_name[np.argmax(result, axis=1)[0]]
        st.success(string)
    else:
        st.write("Please enter some text.")