# Capstone Project: Tendency Towards Bias Application and Sentiment Analyzer

##### Necessary Libraries and Packages contained within `required_libraries_packages.txt` file.

## Problem Statement
- How can bias and sentiment be identified from abstracts and headlines of news articles by utilizing natural language processing, neural networks, and sentiment analyzers? 
- Efficacy will be evaluated based on performance, error metrics, sentiment analysis and recommendations and guidance will be offered. 
- Interested parties in this research include political party affiliates, media outlets, psychologistics and sociologists, and others intrigued by the phenomenon of media influence upon societal thought processes. 

### We present novel methods for determining bias within abstracts/headlines of online articles, label these articles based on relative bias levels, and harness these labels to train a neural network model capable of upscaling this labeling process. Next, we create a custom sentiment analyzer and validate its performance by comparing it to a Valence Aware Dictionary and sEntiment Reasoner (VADER sentiment analyzer) built by a team of professionals. Our processing and analyzing tools are packaged together and available through our Streamlit App titled: Tendency Towards Bias Scoring App. This app runs using our Bias Estimator and Analyzer of Sentiment Tendency (BEAST) Engine. The results of this project will provide the groundwork for future efforts towards understanding bias and sentiment analysis in language processing frameworks. 

### Link Here: https://beastengine.streamlit.app

#### App Instructions: Copy and Paste an Abstract section and Headline from an article. BEAST engine removes filler words and returns a bias score from a neural network model, low magnitude, and a bias score from a labeling function , high magnitude. Click "Show Sentiment Scores" under the text box to see sentiment scores from BEAST and VADER analyzers based on the feelings evoked from the abstract/headline coupling! Afterwards, click "Clear caches and restart," at the bottom of the page, wait until loading finishes, delete the text string and select "Stop" at the top righthand corner of the page to stop the app from running and then the user may submit another abstract and headline! It's that simple!

# Table Of Contents

## See `table_of_contents.md` for Detailed Table of Contents

### `required_libraries_packages.txt`
Contains all required packages and libraries to run project.

### `requirements.txt`
Text document that includes libraries and packages within `streamlit_script.py` - necessary for all streamlit app deployments. 

## 1. Data
- Data will consist of article abstracts and headlines from The New York Times (NYT) webscraped by utilizing NYT API. 
- These will be imported and converted to `Pandas` DataFrames and edited for content. 
- Cleaning and transformation steps will include utilizing Regular Expressions to ensure that the data conform to standards outlined in greater detail within the project notebook. 
- The results of this analysis will serve as input for basic `NLP` models prior to serving as input for more advanced neural network models.

## 2. Modeling

Part I.
- First, a binary classification model will identify whether or not the abstracts and articles came from an `Opinion` piece or other (1 for `Opinion` and 0 for `other`, all other section names).
- Input features for classification model: Abstract and Headline texts.
- Output features for classification model: (1 for `Opinion` and 0 for `other`, all other section names).
Model efficacy is evaluated within notebook (see end of `CleaningEDA_ClassModeling.ipynb` for chosen model implemented in Streamlit app). 
  
Part II. 
  
- Bias scores typical of these modeling projects will be utilized to evaluate the abstracts and headlines for the `Opinion` articles (Cohen Kappa Score generated with help from insights gained from Snorkel labeling functions). After label functions are assessed using Snorkel's built-in metrics, these labeling functions are included in a weighted linear function for labeling abstract/headline concatenated pairs for further neural network modeling. See `Label_Function_Revision_Process.md`, `Label_Functions_Explanation.md`, `Label_Voters_Analogy.md` for general audiences and `Labeling_Functions_Technical.md` for technical audiences. 
  
Part III.
- Model with Long Short Term Memory (LSTM) categorical neural network model, derive sentiment analyzer and compare it with professional VADER sentiment analyzer for performance. Word2Vec variants: Continuous-Bag-Of-Words and and Skip-Gram, as well as Cosine Similarity, are utilized to gain further insights regarding abstract/headline couplings. When we find words that are commonly associated within biased frameworks, we choose these words for labeling functions and go back to Part II for another iteration of labeling functions including these words, survey the Snorkel metrics to see how these performed, and include these words in a new labeling function for another iteration. This outlines the entire development pipeline process for the project. 

## 3. Noteworthy Observations
- The benefits of `NLP` models are that they are not computationally expensive compared with their `NN` counterparts. The drawbacks are that they might not contain as many insights as the latter more complex models.
- The benefits of `NN` models are that they have more hyperparameters and customizability compared with their `NLP` counterparts. The drawbacks are that they may be limited by resources such as hardware and computation times necessary to run optimization steps exhaustively for optimal performance, as was the case for this particular study. 
- Assumptions to confirm / requisite steps to take with independent variables for models:
  - Text data must be cleaned, parsed, and transformed into vectors.

## 4. Resources / References
- `API` Documentation from `NYT`: https://developer.nytimes.com/apis
- Code for webscraping adapted from fellow coursemate's group project, with their permission this code was included in the pipeline for this project.
- Snorkel documentation for Cohen Kappa Score labeling: https://www.snorkel.org
- Pew Center Article on Bias: https://www.pewresearch.org/internet/2017/10/19/the-future-of-truth-and-misinformation-online
- VADER Documentation: https://vadersentiment.readthedocs.io
- Kaggle GPU Documentation: https://www.kaggle.com/code/dansbecker/running-kaggle-kernels-with-a-gpu
- CBOW/Skip Gram: https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa
- CBOW/SKip Gram: https://medium.com/@dube.aditya8/word2vec-skip-gram-cbow-b5e802b00390
- KDE Plot Documentation: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
- Quantum Neural Networks: https://openreview.net/pdf?id=ZLKaNvYFfjd
