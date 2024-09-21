import spacy
from PIL import Image
import streamlit as st
from textblob import TextBlob
from deep_translator import GoogleTranslator

# Define the header text with custom color using HTML
header_html1 = """
    <h1 style="color: darkblue;"><b>Welcome to Sentiment Analysis App</b></h1>
"""

# Render the HTML in Streamlit
st.markdown(header_html1, unsafe_allow_html=True)

image = Image.open("1691232378248.jpeg")
st.image(image, caption='How are you feeling ? Lets check it out')

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize Translator
translator = GoogleTranslator()

# Emotion Classes
class_name = ['Fear', 'Surprise', 'Sadness', 'Joy', 'Love', 'Anger']

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Function to classify sentiment into emotion
def classify_emotion(polarity):
    if polarity < -0.5:
        return 'Fear'
    elif -0.5 <= polarity < -0.2:
        return 'Sadness'
    elif -0.2 <= polarity < 0:
        return 'Anger'
    elif 0 <= polarity < 0.2:
        return 'Surprise'
    elif 0.2 <= polarity < 0.5:
        return 'Love'
    else:
        return 'Joy'

# Function to extract entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Function to translate text to English
def translate_to_english(text, src_lang):
    translated = translator.translate(text, src=src_lang, dest='en')
    return translated

# Function to translate sentiment result back to original language
def translate_to_original(text, target_lang):
    translated = translator.translate(text, src='en', dest=target_lang)
    return translated

# Function to display a progress bar with dynamic color based on polarity
def display_progress_bar(polarity):
    # Normalize the polarity to a range of 0 to 1 for the bar
    normalized_polarity = (polarity + 1) / 2

    # Choose bar color based on polarity value
    if polarity < 0:
        bar_color = 'red'
    else:
        bar_color = 'green'

    # Create custom HTML for the colored progress bar
    bar_html = f"""
    <div style="background-color: lightgray; border-radius: 5px;">
        <div style="width: {normalized_polarity * 100}%; background-color: {bar_color}; height: 20px; border-radius: 5px;"></div>
    </div>
    """
    # Display the custom progress bar using markdown
    st.markdown(bar_html, unsafe_allow_html=True)


# Sidebar for user input
st.sidebar.subheader("Input Options")

# Language selection
selected_lang = st.sidebar.selectbox("Select Input Language", ["en", "es", "fr", "de", "it", "auto"], index=0)

# Subjectivity threshold
subjectivity_threshold = st.sidebar.slider("Subjectivity Threshold", 0.0, 1.0, 0.5)

# Text input on the sidebar
user_input = st.sidebar.text_area("Enter Text Here", "")

# Button to trigger analysis
if st.sidebar.button("Analyze"):
    # Translate text if needed
    original_text = user_input
    if selected_lang != "en" and user_input:
        user_input = translate_to_english(user_input, selected_lang)
        st.sidebar.write(f"Translated Input: {user_input}")

    # Main content layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 style="color: darkblue;"><b>Sentiment & Emotion Analysis</b></h3>', unsafe_allow_html=True)
        if user_input:
            # Sentiment Analysis
            polarity, subjectivity = analyze_sentiment(user_input)
            
            # Classify Emotion
            emotion = classify_emotion(polarity)
            
            # Translate sentiment and emotion to original language
            if selected_lang != "en":
                sentiment_text = translate_to_original(emotion, selected_lang)
            else:
                sentiment_text = emotion

            # Display sentiment and emotion results
            st.write(f"Emotion: {sentiment_text} (Polarity: {polarity:.2f})")
            
            # Display custom progress bar with dynamic color
            display_progress_bar(polarity)
            
            # Subjectivity information
            st.write(f"Subjectivity: {'Subjective' if subjectivity > subjectivity_threshold else 'Objective'} ({subjectivity:.2f})")
        else:
            st.write("Please enter text to analyze.")

    with col2:
        st.markdown('<h3 style="color: darkblue;"><b>Extracted Entities</b></h3>', unsafe_allow_html=True)
        if user_input:
            # Entity Recognition
            entities = extract_entities(user_input)
            
            # Display entities
            if entities:
                for entity, label in entities:
                    st.markdown(f"<span style='background-color: #FFFF00'>{entity}</span> ({label})", unsafe_allow_html=True)
            else:
                st.write("No entities found.")

    # Download results
    st.sidebar.write("**Download Results**")
    if st.sidebar.button("Download"):
        results = f"Emotion: {emotion} (Polarity: {polarity:.2f})\n" \
                  f"Subjectivity: {subjectivity:.2f}\n" \
                  f"Entities: {entities}"
        st.sidebar.download_button("Download Analysis", data=results, file_name="sentiment_emotion_analysis.txt")
