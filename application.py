import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Descargar recursos de NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

# Funciones de preprocesamiento
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_punctuation(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_emojis(text):
    emoji_pattern = re.compile(
        r'['
        u'\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F1E0-\U0001F1FF'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def preprocess_text(input_text):
    cleaned_text = re.sub(r'\n', ' ', input_text)
    cleaned_text = remove_urls(cleaned_text)
    cleaned_text = remove_emojis(cleaned_text)
    cleaned_text = remove_punctuation(cleaned_text)
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def process_tweet_content(tweet):
    cleaned_tweet = preprocess_text(tweet)
    processed_tweet = ' '.join(lemmatizer.lemmatize(word) for word in cleaned_tweet.split() if word not in stop_words_list)
    return processed_tweet

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def remove_specific_words(tweet):
    tweet = normalize_text(tweet)
    tweet = remove_punctuation(tweet).lower()
    cleaned_tweet = ' '.join(word for word in tweet.split() if word not in custom_words)
    return cleaned_tweet

def tokenize_corpus(corpus):
    return tweet_tokenizer.texts_to_sequences(corpus)

def process_input_sentence(input_sentence):
    processed_sentence = process_tweet_content(input_sentence)
    processed_sentence = remove_specific_words(processed_sentence)
    tokenized_sentence = tokenize_corpus([processed_sentence])
    return pad_sequences(tokenized_sentence, maxlen=max_length, padding='post')

# Configuración inicial de Streamlit
st.title("Disaster Tweets Classifier")
st.write("Este modelo predice si un tweet es real o falso en relación a desastres naturales.")

# Cargar modelo LSTM y datos necesarios
model_lstm = tf.keras.models.load_model('model_LSTM23_Final.h5')
train_dataset = pd.read_csv("train.csv", encoding="latin-1")
stop_words_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
custom_words = {'im', 'u', 'â', 'ã', 'one'}
tweet_tokenizer = Tokenizer()
train_tweets = train_dataset['text'].apply(process_tweet_content).values
tweet_tokenizer.fit_on_texts(train_tweets)
max_length = 23

# Entrada de texto
user_input = st.text_input("Ingrese el texto del tweet:")

if st.button("Predecir con LSTM"):
    if user_input:
        processed_text = process_input_sentence(user_input)
        prediction = model_lstm.predict(processed_text)
        prediction_label = "REAL" if prediction >= 0.6 else "FAKE"
        probability = prediction[0][0] if prediction >= 0.6 else 1 - prediction[0][0]
        
        st.subheader("Resultado de la Predicción")
        st.write(f"**Tweet ingresado:** {user_input}")
        st.write(f"**Predicción:** {prediction_label}")
        st.write(f"**Probabilidad:** {probability:.2f}")
    else:
        st.error("Por favor, ingrese un texto para predecir.")
