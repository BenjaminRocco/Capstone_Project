import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler, LabelEncoder
from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.augmentation import transformation_function
from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier
import random
from nltk.corpus import wordnet as wn
from transformers import pipeline

# Load the pre-trained model from the .h5 file
model_path = "model_11_72test.h5"
model = tf.keras.models.load_model(model_path)

# Load RoBERTa for sentiment analysis
# roberta_sentiment_analysis = pipeline("sentiment-analysis", model="roberta-base")

# Load VADER sentiment analyzer
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()


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

@labeling_function()
def lf_keyword_my_binary(x):
    """Return 1 if any of the misleading_bias_terms is present, else return 0."""
    presence = any(term in str(x).lower() for term in misleading_bias_terms)
    return 1 if presence else 0

@labeling_function()
def lf_regex_fake_news_binary(x):
    """Return 1 if any of the bias_words is present, else return 0."""
    presence = any(re.search(fr"\b{word}\b", str(x), flags=re.I) is not None for word in bias_words)
    return 1 if presence else 0

@labeling_function()
def lf_regex_subjective_binary(x):
    """Return 1 if any of the subj_words is present, else return 0."""
    presence = any(re.search(fr"\b{word}\b", str(x), flags=re.I) is not None for word in subj_words)
    return 1 if presence else 0

@labeling_function()
def lf_long_combined_text_binary(text_list):
    """Return 1 if the combined length is greater than 376, else return 0."""
    length = len(" ".join(str(text_list)).split())
    return 1 if length > 376 else 0

@labeling_function()
def lf_textblob_polarity_binary(x):
    """
    We use a third-party sentiment classification model, TextBlob.

    We map the polarity to binary classification: 1 if negative, 0 otherwise.
    """
    polarity = TextBlob(str(x)).sentiment.polarity
    return 1 if polarity < 0 else 0

@labeling_function()
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

# def predict_class(user_input):
#     # 1. Remove non-letters.
#     user_input = re.sub("[^a-zA-Z]", " ", user_input)
    
#     # 2. Convert to lower case, split into individual words.
#     user_input = user_input.lower().split()
#     # Tokenize and pad the input sequence

#     tokenizer.fit_on_texts([user_input])
#     sequence = tokenizer.texts_to_sequences([user_input])
#     padded_sequence = pad_sequences(sequence, maxlen=max_length)

#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     word_tokens = word_tokenize(" ".join(user_input))
#     filtered_tokens = [word for word in word_tokens if word.lower() not in stop_words]

#     # Print the phrase with stopwords removed
#     filtered_phrase = " ".join(filtered_tokens)
#     st.write(f"Phrase with stopwords removed: {filtered_phrase}")

#     # Make a prediction using the neural net model
#     model_score = np.argmax(model.predict(padded_sequence))/10
#     st.write(f"Neural Net Model Score: {model_score:.2f}")

#     # Call the previous function to display outcomes of labeling functions
#     st.subheader("Labeling Function Outcomes:")
#     lf_outcomes(" ".join(filtered_tokens), model_score)

# Function to predict and display outcomes
def predict_and_display_outcomes(user_input, show_sentiment_scores):
    # Tokenize and pad the input sequence
    tokenizer.fit_on_texts([user_input])
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)

    # Display phrase with stopwords removed
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(user_input)
    filtered_tokens = [word for word in word_tokens if word.lower() not in stop_words]
    filtered_phrase = " ".join(filtered_tokens)
    st.write(f"Phrase with stopwords removed: {filtered_phrase}")

    # Make a prediction using the neural net model
    model_score = np.argmax(model.predict(padded_sequence)) / 10
    st.write(f"Neural Net Model Score: {model_score:.2f}")

    # Use RoBERTa for sentiment analysis
    # roberta_score = roberta_sentiment_analysis(user_input)[0]['score']
    # st.write(f"RoBERTa Sentiment Score: {roberta_score:.2f}")

    # Use VADER sentiment analyzer
    vader_score = vader_analyzer.polarity_scores(user_input)['compound']
    st.write(f"VADER Sentiment Score: {vader_score:.2f}")

    # Call the previous function to display outcomes of labeling functions
    st.subheader("Labeling Function Outcomes:")
    lf_outcomes(user_input, model_score)

    combined_score = combined_binary_bias_score(user_input)
    pred_sent_score = combined_score + model_score

    if vader_score < 0:
        pred_sent_score *= -1
    elif vader_score > 0:
        pred_sent_score *= 1


    # Display sentiment scores if checkbox is selected
    if show_sentiment_scores:
        st.subheader("Sentiment Scores:")
        # st.write("RoBERTa Sentiment Score:", roberta_score)
        st.write("Predicted Sentiment Score:", pred_sent_score)
        st.write("VADER Sentiment Score:", vader_score)

# Define a function to display outcomes of labeling functions
@st.cache_resource()
def lf_outcomes(user_input, model_score):
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
    st.write(f"Total Tendency Towards Bias Score: {combined_score:.2f}")

# Streamlit app
def main():
    st.title("Tendency Towards Bias Scoring App")

    # User input
    user_input = st.text_area("Enter a combined abstract/headline string:")

    # Checkbox for displaying sentiment scores
    show_sentiment_scores = st.checkbox("Show Sentiment Scores")

    if st.button("Predict"):
        if user_input:
            # Call the predict_and_display_outcomes function
            predict_and_display_outcomes(user_input, show_sentiment_scores)

# Run the Streamlit app
if __name__ == "__main__":
    main()