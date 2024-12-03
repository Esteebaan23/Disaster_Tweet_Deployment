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

def remove_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_punctuation(text):
    """Remove punctuation from the text."""
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_emojis(text):
    """Remove emojis from the text."""
    emoji_pattern = re.compile(
        r'['
        u'\U0001F600-\U0001F64F'  # Emoticons
        u'\U0001F300-\U0001F5FF'  # Symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # Transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # Flags (iOS)
        u'\U00002702-\U000027B0'  # Miscellaneous symbols
        u'\U000024C2-\U0001F251'  # Other symbols
        ']+',
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def remove_html_tags(text):
    """Remove HTML tags from the text."""
    html_pattern = re.compile(r'<.*?>|&(?:[a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return html_pattern.sub('', text)

def preprocess_text(input_text):
    """Clean and preprocess the input text."""
    cleaned_text = str(input_text)
    cleaned_text = re.sub(r"\bI'm\b", "I am", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\byou're\b", "you are", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bthey're\b", "they are", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bcan't\b", "cannot", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bwon't\b", "will not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bdon't\b", "do not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bdoesn't\b", "does not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bain't\b", "am not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bwe're\b", "we are", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bit's\b", "it is", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bu\b", "you", cleaned_text, flags=re.IGNORECASE)

    cleaned_text = re.sub(r"&gt;", ">", cleaned_text)
    cleaned_text = re.sub(r"&lt;", "<", cleaned_text)
    cleaned_text = re.sub(r"&amp;", "&", cleaned_text)


    cleaned_text = re.sub(r"\bw/\b", "with", cleaned_text)  # "w/" → "with"
    cleaned_text = re.sub(r"\blmao\b", "laughing my ass off", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"<3", "love", cleaned_text)  # Corazón → "love"
    cleaned_text = re.sub(r"\bph0tos\b", "photos", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bamirite\b", "am I right", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\btrfc\b", "traffic", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b16yr\b", "16 year", cleaned_text)

    cleaned_text = str(cleaned_text).lower()  # Convert to lowercase

    # Remove unwanted patterns
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)  # Remove content inside brackets
    cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)  # Remove URLs
    cleaned_text = re.sub(r'<.*?>+', '', cleaned_text)  # Remove HTML tags
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)  # Replace newlines with spaces
    cleaned_text = re.sub(r'\w*\d\w*', '', cleaned_text)  # Remove words with numbers

    # Call additional cleaning functions
    cleaned_text = remove_urls(cleaned_text)  # Remove URLs
    cleaned_text = remove_emojis(cleaned_text)  # Remove emojis
    cleaned_text = remove_html_tags(cleaned_text)  # Remove HTML tags
    cleaned_text = remove_punctuation(cleaned_text)  # Remove punctuation

    return cleaned_text

def process_tweet_content(tweet):
    cleaned_tweet = preprocess_text(tweet)
    processed_tweet = ' '.join(lemmatizer.lemmatize(word) for word in cleaned_tweet.split() if word not in stop_words_list)

    return processed_tweet

def normalize_text(text):
    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return normalized_text

def remove_specific_words(tweet):
    tweet = normalize_text(tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation)).lower()
    cleaned_tweet = ' '.join(word for word in tweet.split() if word not in custom_words)

    return cleaned_tweet

def tokenize_corpus(corpus):
    return tweet_tokenizer.texts_to_sequences(corpus)


def process_input_sentence(input_sentence):
    processed_sentence = process_tweet_content(input_sentence)
    processed_sentence = process_tweet_content(processed_sentence)
    #print(processed_sentence)
    processed_sentence = remove_specific_words(processed_sentence)
    #print(processed_sentence)
    tokenized_sentence = tokenize_corpus([processed_sentence])
    #print(tokenized_sentence)
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=max_length, padding='post')
    #print(padded_sentence)
    return padded_sentence

# Configuración inicial de Streamlit
st.title("Disaster Tweets Classifier")
st.write("This model predicts whether a tweet is real or fake in relation to natural disasters.")

# Cargar modelo LSTM y datos necesarios
model_lstm = tf.keras.models.load_model('model_LSTM23_Final.h5')
train_dataset = pd.read_csv("train.csv", encoding="latin-1")
stop_words_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
extra_stop_words = ['u', 'im', 'r']
extra_stop_words2 = [
    'u', 'im', 'r', 'ur', 'pls', 'thx',
    'b4', 'omw', 'ppl', 'msg', 'lvl',
    'sos', '911', 'help', 'asap',
    'wtf', 'omg', 'idk', 'nvm',
    'brb', 'btw', 'lmk', 'imo',
    'stay', 'safe', 'evacuate', 'fyi'
]
all_stop_words = stop_words_list + extra_stop_words + extra_stop_words2
train_dataset['text_clean'] = train_dataset['text'].apply(process_tweet_content)
custom_words = {'im', 'u','â', 'ã', 'one', 'ã ã', 'ã ã', 'ã âª', 'ã â', 'â ã', 'â', 'âª','â', 'ã','aaa', 'rt','aa', 'ye'}
train_dataset['text_clean'] = train_dataset['text_clean'].apply(remove_specific_words)

train_tweets = train_dataset['text_clean'].values
tweet_tokenizer = Tokenizer()
tweet_tokenizer.fit_on_texts(train_tweets)
vocabulary_size = len(tweet_tokenizer.word_index) + 1
max_length = 23




user_input = st.text_input("Ingrese el texto del tweet:")

if st.button("Predict"):
    if user_input:
        processed_text = process_input_sentence(user_input)
        prediction = model_lstm.predict(processed_text)
        prediction_label = "REAL" if prediction >= 0.6 else "FAKE"
        probability = prediction[0][0] if prediction >= 0.6 else 1 - prediction[0][0]
        
        st.subheader("Resultado de la Predicción")
        st.write(f"**Tweet entered:** {user_input}")
        st.write(f"**Prediction:** {prediction_label}")
        st.write(f"**Probability:** {probability:.2f}")
    else:
        st.error("Please enter a text to predict.")
