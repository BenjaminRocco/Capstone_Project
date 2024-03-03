import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
import re
import os #
from textblob import TextBlob

# Load the pre-trained model from the .h5 file
# model_path = "/kaggle/input/model_11_72test/keras/model_11/1/model_11_72test.keras"
# model = load_model(rmodel_path)
model_path = "model_11_72test.h5"
model = tf.keras.models.load_model(model_path)

# Define variables
vocab_size = 23100
embedding_dim = 100
max_length = 78
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

misleading_bias_terms = ['trump', 'u', 'america', 'american', 'new', 'people', 'states', 'president', 'many', 'states', 'united', 'americans', 'one']
bias_words = ['fake', 'news', 'fale', 'biased', 'unreliable', 'propaganda', 'misleading', 'partisan', 'manipulative']
subj_words = ['feel', 'feels', 'thinks', 'thought', 'thoughts', 'opinion', 'bias', 'think','felt', 'believe', 'believed','believes','believer']

# @labeling_function()
def lf_keyword_my_binary(x):
    """Return 1 if any of the misleading_bias_terms is present, else return 0."""
    presence = any(term in str(x).lower() for term in misleading_bias_terms)
    return 1 if presence else 0

# @labeling_function()
def lf_regex_fake_news_binary(x):
    """Return 1 if any of the bias_words is present, else return 0."""
    presence = any(re.search(fr"\b{word}\b", str(x), flags=re.I) is not None for word in bias_words)
    return 1 if presence else 0

# @labeling_function()
def lf_regex_subjective_binary(x):
    """Return 1 if any of the subj_words is present, else return 0."""
    presence = any(re.search(fr"\b{word}\b", str(x), flags=re.I) is not None for word in subj_words)
    return 1 if presence else 0

# @labeling_function()
def lf_long_combined_text_binary(text_list):
    """Return 1 if the combined length is greater than 376, else return 0."""
    length = len(" ".join(str(text_list)).split())
    return 1 if length > 376 else 0

# @labeling_function()
def lf_textblob_polarity_binary(x):
    """
    We use a third-party sentiment classification model, TextBlob.

    We map the polarity to binary classification: 1 if negative, 0 otherwise.
    """
    polarity = TextBlob(str(x)).sentiment.polarity
    return 1 if polarity < 0 else 0

# @labeling_function()
def lf_textblob_subjectivity_binary(x):
    """
    We use a third-party sentiment classification model, TextBlob.

    We map the subjectivity to binary classification: 1 if high subjectivity, 0 otherwise.
    """
    subjectivity = TextBlob(str(x)).sentiment.subjectivity
    return 1 if subjectivity > 0.5 else 0

# Define weights for each binary labeling function
weight_lf_keyword_my_binary = 0.2
weight_lf_regex_fake_news_binary = 0.1
weight_lf_regex_subjective_binary = 0.1
weight_lf_long_combined_text_binary = 0.2
weight_lf_textblob_polarity_binary = 0.1
weight_lf_textblob_subjectivity_binary = 0.3

# @labeling_function()
def combined_binary_bias_score(x):
    """Combine binary labeling functions into a linear equation."""
    lf1_score = lf_keyword_my_binary(x) * weight_lf_keyword_my_binary
    lf2_score = lf_regex_fake_news_binary(x) * weight_lf_regex_fake_news_binary
    lf3_score = lf_regex_subjective_binary(x) * weight_lf_regex_subjective_binary
    lf4_score = lf_long_combined_text_binary(x) * weight_lf_long_combined_text_binary
    lf5_score = lf_textblob_polarity_binary(x) * weight_lf_textblob_polarity_binary
    lf6_score = lf_textblob_subjectivity_binary(x) * weight_lf_textblob_subjectivity_binary

    # Combine scores with weights
    combined_score = lf1_score + lf2_score + lf3_score + lf4_score + lf5_score + lf6_score

    # Normalize to the range [0, 1]
    normalized_score = max(0, min(combined_score, 1))

    return normalized_score

def predict_class(user_input):
    # Tokenize and pad the input sequence
    tokenizer.fit_on_texts([user_input])
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)

    # Make a prediction using the neural net model
    model_score = model.predict(padded_sequence)[0][0]
    st.write(f"Neural Net Model Score: {model_score:.2f}")

    # Call the previous function to display outcomes of labeling functions
    st.subheader("Labeling Function Outcomes:")
    lf_outcomes(user_input)

# Define a function to display outcomes of labeling functions
@st.cache
def lf_outcomes(user_input):
    # Labeling Function 1
    lf1_outcome = lf_keyword_my_binary(user_input)
    st.write(f"LF 1 - Keyword My Binary: Outcome - {lf1_outcome}, Score - {lf1_outcome * weight_lf_keyword_my_binary:.2f}")

    # Labeling Function 2
    lf2_outcome = lf_regex_fake_news_binary(user_input)
    st.write(f"LF 2 - Regex Fake News Binary: Outcome - {lf2_outcome}, Score - {lf2_outcome * weight_lf_regex_fake_news_binary:.2f}")
    
    # Labeling Function 3
    lf3_outcome = lf_regex_subjective_binary(user_input)
    st.write(f"LF 3 - Regex Subjective Binary: Outcome - {lf3_outcome}, Score - {lf3_outcome * weight_lf_regex_subjective_binary:.2f}")
    
    # Labeling Function 4
    lf4_outcome = lf_long_combined_text_binary(user_input)
    st.write(f"LF 4 - Long Combined Text Binary: Outcome - {lf4_outcome}, Score - {lf4_outcome * weight_lf_long_combined_text_binary:.2f}")
    
    # Labeling Function 5
    lf5_outcome = lf_textblob_polarity_binary(user_input)
    st.write(f"LF 5 - Textblob Polarity Binary: Outcome - {lf5_outcome}, Score - {lf5_outcome * weight_lf_textblob_polarity_binary:.2f}")
    
    # Labeling Function 6
    lf6_outcome = lf_textblob_subjectivity_binary(user_input)
    st.write(f"LF 6 - Textblob Subjective Binary: Outcome - {lf6_outcome}, Score - {lf6_outcome * weight_lf_textblob_subjectivity_binary:.2f}")

    # Combined Binary Bias Score
    combined_score = combined_binary_bias_score(user_input)
    st.write(f"Total Bias Score: {combined_score:.2f}")

# Streamlit app
def main():
    st.title("Bias Scoring App")

    # User input
    user_input = st.text_area("Enter a combined abstract/headline string:")

    if user_input:
        # Call the predict_class function
        predict_class(user_input)

# Run the Streamlit app
if __name__ == "__main__":
    main()