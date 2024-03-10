import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

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
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from transformers import pipeline
import pickle
import keras
from keras.models import load_model

# Load the pre-trained model from the .h5 file
# model_path = "model_11_72test.h5" # Best Performing - Currently Set
# model = load_model(model_path, custom_objects={'Embedding': keras.layers.Embedding}, compile=False)
model_path = "model_11.keras" # Best Performing - Currently Set
model = tf.keras.models.load_model(model_path)
# config = model.get_config()
# st.write(f"{config}")

# Insert your relative path here
model_filepath = 'binary_classification_SVCTVEC.pkl'

# Load the model using pickle
with open(model_filepath, 'rb') as model_file:
    loaded__bin_model = pickle.load(model_file)

# Load VADER sentiment analyzer
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
# Neg words
past_tense_keywords = ["hurt", "blamed", "harmed", "accused"]
present_tense_keywords = ["hurts", "blames", "harms", "accuses"]
active_voice_keywords = ["hurting", "blaming", "harming", "accusing"]

# Keyword binary functions
keywords = [
    "maps", "county", "election", "coronavirus", "case",
    "risk", "cases", "covid", "latest", "trump",
    "ukraine", "russia", "war", "reminiscent",
    "removes", "proceed", "ponder"
]

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
    return 1 if length > 133 else 0 

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
    # Check if either of the two phrases occurs in the text
    if "see full results maps" in str(x).lower() or "see latest charts maps" in str(x).lower():
        return 0
    
    # Calculate subjectivity using TextBlob
    subjectivity = TextBlob(str(x)).sentiment.subjectivity
    
    # Return 1 if high subjectivity, 0 otherwise
    return 1 if subjectivity > 0.5 else 0

@labeling_function()
def lf_past_tense_keywords_binary(x):
    """Return BIAS if any of the subj_words is present, else ABSTAIN."""
    presence = any(re.search(fr"\b{word}\b", str(x), flags=re.I) is not None for word in past_tense_keywords)
    return 1 if presence else 0

@labeling_function()
def lf_present_tense_keywords_binary(x):
    """Return BIAS if any of the subj_words is present, else ABSTAIN."""
    presence = any(re.search(fr"\b{word}\b", str(x), flags=re.I) is not None for word in present_tense_keywords)
    return 1 if presence else 0

@labeling_function()
def lf_active_voice_keywords_binary(x):
    """Return BIAS if any of the subj_words is present, else ABSTAIN."""
    presence = any(re.search(fr"\b{word}\b", str(x), flags=re.I) is not None for word in active_voice_keywords)
    return 1 if presence else 0

keywords = [
    "maps", "county", "election", "coronavirus", "case",
    "risk", "cases", "covid", "latest", "trump",
    "ukraine", "russia", "war", "reminiscent",
    "removes", "proceed", "ponder"
]

keywords_pattern = "|".join(fr"\b{re.escape(keyword)}\b" for keyword in keywords)

@labeling_function()
def lf_keyword_maps_binary(x):
    return 1 if re.search(fr"\b{re.escape('maps')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_county_binary(x):
    return 1 if re.search(fr"\b{re.escape('county')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_election_binary(x):
    return 1 if re.search(fr"\b{re.escape('election')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_coronavirus_binary(x):
    return 1 if re.search(fr"\b{re.escape('coronavirus')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_case_binary(x):
    return 1 if re.search(fr"\b{re.escape('case')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_risk_binary(x):
    return 1 if re.search(fr"\b{re.escape('risk')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_cases_binary(x):
    return 1 if re.search(fr"\b{re.escape('cases')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_covid_binary(x):
    return 1 if re.search(fr"\b{re.escape('covid')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_latest_binary(x):
    return 1 if re.search(fr"\b{re.escape('latest')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_trump_binary(x):
    return 1 if re.search(fr"\b{re.escape('trump')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_ukraine_binary(x):
    return 1 if re.search(fr"\b{re.escape('ukraine')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_russia_binary(x):
    return 1 if re.search(fr"\b{re.escape('russia')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_war_binary(x):
    return 1 if re.search(fr"\b{re.escape('war')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_reminiscent_binary(x):
    return 1 if re.search(fr"\b{re.escape('reminiscent')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_removes_binary(x):
    return 1 if re.search(fr"\b{re.escape('removes')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_proceed_binary(x):
    return 1 if re.search(fr"\b{re.escape('proceed')}\b", str(x), flags=re.I) else 0

@labeling_function()
def lf_keyword_ponder_binary(x):
    return 1 if re.search(fr"\b{re.escape('ponder')}\b", str(x), flags=re.I) else 0

# Define weights for each binary labeling function
weight_lf_keyword_my_binary = 0.1 # Lower this to 0.05 with neg words, raise to 0.2 without neg words move to 0.01 with other keywords
weight_lf_regex_fake_news_binary = 0.1
weight_lf_regex_subjective_binary = 0.1
weight_lf_long_combined_text_binary = 0.1
weight_lf_textblob_polarity_binary = 0.2 
weight_lf_textblob_subjectivity_binary = 0.4 # Up to 0.39 when individual keywords used
# Neg/pos words - Set weights to 0.00 for original model and put to 0.05 for subsequent pos and neg word tests
weight_lf_past_tense_keywords_binary = 0.00 
weight_lf_present_tense_keywords_binary = 0.00
weight_lf_active_voice_keywords_binary = 0.00
# sub 0.1
weight_lf_keyword_maps_binary = -0.01
weight_lf_keyword_county_binary = -0.01
weight_lf_keyword_election_binary = -0.01
weight_lf_keyword_coronavirus_binary = -0.01
weight_lf_keyword_case_binary = -0.01
weight_lf_keyword_risk_binary = -0.01
weight_lf_keyword_cases_binary = -0.01
weight_lf_keyword_covid_binary = -0.02
weight_lf_keyword_latest_binary = -0.01
# add 0.1
weight_lf_keyword_trump_binary = 0.03
weight_lf_keyword_ukraine_binary = 0.01
weight_lf_keyword_russia_binary = 0.01
weight_lf_keyword_war_binary = 0.01
weight_lf_keyword_reminiscent_binary = 0.01
weight_lf_keyword_removes_binary = 0.01
weight_lf_keyword_proceed_binary = 0.01
weight_lf_keyword_ponder_binary = 0.01

def combined_binary_bias_score(x):
    """Combine binary labeling functions into a linear equation."""
    lf1_score = lf_keyword_my_binary(x) * weight_lf_keyword_my_binary
    lf2_score = lf_regex_fake_news_binary(x) * weight_lf_regex_fake_news_binary
    lf3_score = lf_regex_subjective_binary(x) * weight_lf_regex_subjective_binary
    lf4_score = lf_long_combined_text_binary(x) * weight_lf_long_combined_text_binary
    lf5_score = lf_textblob_polarity_binary(x) * weight_lf_textblob_polarity_binary
    lf6_score = lf_textblob_subjectivity_binary(x) * weight_lf_textblob_subjectivity_binary
    # neg words
    lf7_score = lf_past_tense_keywords_binary(x) * weight_lf_past_tense_keywords_binary
    lf8_score = lf_present_tense_keywords_binary(x) * weight_lf_present_tense_keywords_binary
    lf9_score = lf_active_voice_keywords_binary(x) * weight_lf_active_voice_keywords_binary
    # keyword binary functions
    lf10_score = lf_keyword_maps_binary(x) * weight_lf_keyword_maps_binary
    lf11_score = lf_keyword_county_binary(x) * weight_lf_keyword_county_binary
    lf12_score = lf_keyword_election_binary(x) * weight_lf_keyword_election_binary
    lf13_score = lf_keyword_coronavirus_binary(x) * weight_lf_keyword_coronavirus_binary
    lf14_score = lf_keyword_case_binary(x) * weight_lf_keyword_case_binary
    lf15_score = lf_keyword_risk_binary(x) * weight_lf_keyword_risk_binary
    lf16_score = lf_keyword_cases_binary(x) * weight_lf_keyword_cases_binary
    lf17_score = lf_keyword_covid_binary(x) * weight_lf_keyword_covid_binary
    lf18_score = lf_keyword_latest_binary(x) * weight_lf_keyword_latest_binary
    lf19_score = lf_keyword_trump_binary(x) * weight_lf_keyword_trump_binary
    lf20_score = lf_keyword_ukraine_binary(x) * weight_lf_keyword_ukraine_binary
    lf21_score = lf_keyword_russia_binary(x) * weight_lf_keyword_russia_binary
    lf22_score = lf_keyword_war_binary(x) * weight_lf_keyword_war_binary
    lf23_score = lf_keyword_reminiscent_binary(x) * weight_lf_keyword_reminiscent_binary
    lf24_score = lf_keyword_removes_binary(x) * weight_lf_keyword_removes_binary
    lf25_score = lf_keyword_proceed_binary(x) * weight_lf_keyword_proceed_binary
    lf26_score = lf_keyword_ponder_binary(x) * weight_lf_keyword_ponder_binary

    # Combine scores with weights
    combined_score = (
            lf1_score + lf2_score + lf3_score + lf4_score +
            lf5_score + lf6_score + lf7_score + lf8_score + lf9_score +
            lf10_score + lf11_score + lf12_score + lf13_score +
            lf14_score + lf15_score + lf16_score + lf17_score +
            lf18_score + lf19_score + lf20_score + lf21_score +
            lf22_score + lf23_score + lf24_score + lf25_score + lf26_score
    )
    # Normalize to the range [0, 1]
    normalized_score = max(0, min(combined_score, 1))

    return normalized_score

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
    st.markdown(f"**Phrase with stopwords removed:**\n*{filtered_phrase}*")

    binary_class_model_score = loaded__bin_model.predict([user_input]).item()

    # Display result based on the binary model prediction - everyone outputting in reverse so 0 is chosen for true return (opinion)
    if binary_class_model_score == 0:
        st.markdown("**This abstract/headline coupling came from an opinion section.**")
    else:
        st.markdown("**This abstract/headline coupling came from a non-opinion section.**")

    st.subheader("**Neural Network Outcomes:**")

    # Make a prediction using the neural net model
    model_score = np.argmax(model.predict(padded_sequence)) / 10
    st.markdown(f"**Neural Net Model Tendency Towards Bias Score:**\n*{model_score:.2f}.*")

    if 0.0 <= model_score <= 0.20:
        st.write("**Bias Tier:**\n*None to Slight Bias*")
    elif 0.21 <= model_score <= 0.40:
        st.write("**Bias Tier:**\n*Fair Bias*")
    elif 0.41 <= model_score <= 0.60:
        st.write("**Bias Tier:**\n*Moderate Bias*")
    elif 0.61 <= model_score <= 0.80:
        st.write("**Bias Tier:**\n*Substantial Bias*")
    elif 0.81 <= model_score <= 1.00:
        st.write("**Bias Tier:**\n*Perfectly Bias*")

    # Use VADER sentiment analyzer
    vader_score_total = vader_analyzer.polarity_scores(user_input)['compound']
    vader_score_pos = vader_analyzer.polarity_scores(user_input)['pos']
    vader_score_neutral = vader_analyzer.polarity_scores(user_input)['neu']
    vader_score_neg = vader_analyzer.polarity_scores(user_input)['neg']

    # Call the previous function to display outcomes of labeling functions
    st.subheader("**Labeling Function Outcomes:**")
    lf_outcomes(user_input, model_score)

    combined_score = combined_binary_bias_score(user_input)
    # This sentiment score combines the predicted model score and combined train score, where train tends more towards bias and predicted errs on less bias
    # Their sum is a total that captures sentiment: closer to positive 1 indicates more positive sentiment whereas closer to negative 1 indicates more negative sentiment.
    sent_score = combined_score + model_score

    if vader_score_total < 0:
        sent_score *= -1
    elif vader_score_total > 0:
        sent_score *= 1

    # Display sentiment scores if checkbox is selected
    if show_sentiment_scores:
        st.header("**Sentiment Scores:**")
        st.markdown(f"**Model Sentiment Score:**\n*{sent_score:.4f}.*")
        st.markdown(f"**VADER Total Sentiment Score:**\n*{vader_score_total:.4f}.*")
        st.markdown(f"**VADER Positive Score:**\n*{vader_score_pos:.4f}.*")
        st.markdown(f"**VADER Neutral Score:**\n*{vader_score_neutral:.4f}.*")
        st.markdown(f"**VADER Negative Score:**\n*{vader_score_neg:.4f}.*")

    st.subheader("**All results generated by Bias Estimator and Analyzer of Sentiment Tendency (BEAST) Engine**")


# Define a function to display outcomes of labeling functions
@st.cache_resource()
def lf_outcomes(user_input, model_score):
    # Labeling Function 1
    # lf1_outcome = lf_keyword_my_binary(user_input)
    # st.write(f"LF 1 - Keyword My Binary: Outcome - {lf1_outcome}, Score - {lf1_outcome * weight_lf_keyword_my_binary:.2f}.")

    # # Labeling Function 2
    # lf2_outcome = lf_regex_fake_news_binary(user_input)
    # st.write(f"LF 2 - Regex Fake News Binary: Outcome - {lf2_outcome}, Score - {lf2_outcome * weight_lf_regex_fake_news_binary:.2f}.")

    # # Labeling Function 3
    # lf3_outcome = lf_regex_subjective_binary(user_input)
    # st.write(f"LF 3 - Regex Subjective Binary: Outcome - {lf3_outcome}, Score - {lf3_outcome * weight_lf_regex_subjective_binary:.2f}.")

    # # Labeling Function 4
    # lf4_outcome = lf_long_combined_text_binary(user_input)
    # st.write(f"LF 4 - Long Combined Text Binary: Outcome - {lf4_outcome}, Score - {lf4_outcome * weight_lf_long_combined_text_binary:.2f}.")

    # # Labeling Function 5
    # lf5_outcome = lf_textblob_polarity_binary(user_input)
    # st.write(f"LF 5 - Textblob Polarity Binary: Outcome - {lf5_outcome}, Score - {lf5_outcome * weight_lf_textblob_polarity_binary:.2f}.")

    # # Labeling Function 6
    # lf6_outcome = lf_textblob_subjectivity_binary(user_input)
    # st.write(f"LF 6 - Textblob Subjective Binary: Outcome - {lf6_outcome}, Score - {lf6_outcome * weight_lf_textblob_subjectivity_binary:.2f}.")

    # # Labeling Function 7
    # lf7_outcome = lf_past_tense_keywords_binary(user_input)
    # st.write(f"LF 7 - Past Tense Keywords Binary: Outcome - {lf7_outcome}, Score - {lf7_outcome * weight_lf_past_tense_keywords_binary:.2f}.")

    # # Labeling Function 8
    # lf8_outcome = lf_present_tense_keywords_binary(user_input)
    # st.write(f"LF 8 - Present Tense Keywords Binary: Outcome - {lf8_outcome}, Score - {lf8_outcome * weight_lf_present_tense_keywords_binary:.2f}.")

    # # Labeling Function 9
    # lf9_outcome = lf_active_voice_keywords_binary(user_input)
    # st.write(f"LF 9 - Active Voice Keywords Binary: Outcome - {lf9_outcome}, Score - {lf9_outcome * weight_lf_active_voice_keywords_binary:.2f}.")

    # # Labeling Function 10
    # lf10_outcome = lf_keyword_maps_binary(user_input)
    # st.write(f"LF 10 - Keyword Maps Binary: Outcome - {lf10_outcome}, Score - {lf10_outcome * weight_lf_keyword_maps_binary:.2f}.")

    # # Labeling Function 11
    # lf11_outcome = lf_keyword_county_binary(user_input)
    # st.write(f"LF 11 - Keyword County Binary: Outcome - {lf11_outcome}, Score - {lf11_outcome * weight_lf_keyword_county_binary:.2f}.")

    # # Labeling Function 12
    # lf12_outcome = lf_keyword_election_binary(user_input)
    # st.write(f"LF 12 - Keyword Election Binary: Outcome - {lf12_outcome}, Score - {lf12_outcome * weight_lf_keyword_election_binary:.2f}.")

    # # Labeling Function 13
    # lf13_outcome = lf_keyword_coronavirus_binary(user_input)
    # st.write(f"LF 13 - Keyword Coronavirus Binary: Outcome - {lf13_outcome}, Score - {lf13_outcome * weight_lf_keyword_coronavirus_binary:.2f}.")

    # # Labeling Function 14
    # lf14_outcome = lf_keyword_case_binary(user_input)
    # st.write(f"LF 14 - Keyword Case Binary: Outcome - {lf14_outcome}, Score - {lf14_outcome * weight_lf_keyword_case_binary:.2f}.")

    # # Labeling Function 15
    # lf15_outcome = lf_keyword_risk_binary(user_input)
    # st.write(f"LF 15 - Keyword Risk Binary: Outcome - {lf15_outcome}, Score - {lf15_outcome * weight_lf_keyword_risk_binary:.2f}.")

    # # Labeling Function 16
    # lf16_outcome = lf_keyword_cases_binary(user_input)
    # st.write(f"LF 16 - Keyword Cases Binary: Outcome - {lf16_outcome}, Score - {lf16_outcome * weight_lf_keyword_cases_binary:.2f}.")

    # # Labeling Function 17
    # lf17_outcome = lf_keyword_covid_binary(user_input)
    # st.write(f"LF 17 - Keyword COVID Binary: Outcome - {lf17_outcome}, Score - {lf17_outcome * weight_lf_keyword_covid_binary:.2f}.")

    # # Labeling Function 18
    # lf18_outcome = lf_keyword_latest_binary(user_input)
    # st.write(f"LF 18 - Keyword Latest Binary: Outcome - {lf18_outcome}, Score - {lf18_outcome * weight_lf_keyword_latest_binary:.2f}.")

    # # Labeling Function 19
    # lf19_outcome = lf_keyword_trump_binary(user_input)
    # st.write(f"LF 19 - Keyword Trump Binary: Outcome - {lf19_outcome}, Score - {lf19_outcome * weight_lf_keyword_trump_binary:.2f}.")

    # # Labeling Function 20
    # lf20_outcome = lf_keyword_ukraine_binary(user_input)
    # st.write(f"LF 20 - Keyword Ukraine Binary: Outcome - {lf20_outcome}, Score - {lf20_outcome * weight_lf_keyword_ukraine_binary:.2f}.")

    # # Labeling Function 21
    # lf21_outcome = lf_keyword_russia_binary(user_input)
    # st.write(f"LF 21 - Keyword Russia Binary: Outcome - {lf21_outcome}, Score - {lf21_outcome * weight_lf_keyword_russia_binary:.2f}.")

    # # Labeling Function 22
    # lf22_outcome = lf_keyword_war_binary(user_input)
    # st.write(f"LF 22 - Keyword War Binary: Outcome - {lf22_outcome}, Score - {lf22_outcome * weight_lf_keyword_war_binary:.2f}.")

    # # Labeling Function 23
    # lf23_outcome = lf_keyword_reminiscent_binary(user_input)
    # st.write(f"LF 23 - Keyword Reminiscent Binary: Outcome - {lf23_outcome}, Score - {lf23_outcome * weight_lf_keyword_reminiscent_binary:.2f}.")

    # # Labeling Function 24
    # lf24_outcome = lf_keyword_removes_binary(user_input)
    # st.write(f"LF 24 - Keyword Removes Binary: Outcome - {lf24_outcome}, Score - {lf24_outcome * weight_lf_keyword_removes_binary:.2f}.")

    # # Labeling Function 25
    # lf25_outcome = lf_keyword_proceed_binary(user_input)
    # st.write(f"LF 25 - Keyword Proceed Binary: Outcome - {lf25_outcome}, Score - {lf25_outcome * weight_lf_keyword_proceed_binary:.2f}.")

    # # Labeling Function 26
    # lf26_outcome = lf_keyword_ponder_binary(user_input)
    # st.write(f"LF 26 - Keyword Ponder Binary: Outcome - {lf26_outcome}, Score - {lf26_outcome * weight_lf_keyword_ponder_binary:.2f}.")

    # Combined Binary Bias Score
    combined_score = combined_binary_bias_score(user_input)
    st.write(f"**Labeling Function Tendency Towards Bias Score:**\n*{combined_score:.2f}.*")

    if 0.0 <= combined_score <= 0.20:
        st.write("**Bias Tier:**\n*None to Slight Bias*")
    elif 0.21 <= combined_score <= 0.40:
        st.write("**Bias Tier:**\n*Fair Bias*")
    elif 0.41 <= combined_score <= 0.60:
        st.write("**Bias Tier:**\n*Moderate Bias*")
    elif 0.61 <= combined_score <= 0.80:
        st.write("**Bias Tier:**\n*Substantial Bias*")
    elif 0.81 <= combined_score <= 1.00:
        st.write("**Bias Tier:**\n*Perfectly Bias*")

# Streamlit app
def main():
    st.title("Tendency Towards Bias Scoring App")

    st.subheader("**Powered by BEAST Engine**")

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