## import libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

#load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

#load the pretrained model
model = load_model('simp_rnn_imdb.h5')

## function to decode review
def decode_reveiw(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

## function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment, prediction[0][0]


## streamlit app
st.title('IMDB movie review sentiment analysis')
st.write('Enter a movie review to classify it as positive or negative')

#user input
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    
    # make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review')