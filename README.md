# Capstone Project: Tendency Towards Bias Application and Sentiment Analyzer

##### For Necessary Libraries and Packages please see `required_libraries_packages.txt` file.

## Table Of Contents

### Overview
`README.md`

### Code
`part_01`
`part_02`
`part_03`

### Data
`Data`

### Documents 
`Documentation`

### streamlit_script.py
Initial Streamlit application script

### streamlit_script_expand.py
Final iteration of Streamlit application script - Deployment Version

### model_11%%%
Long Short Term Memory (LSTM) Neural Network (NN) models

### required_libraries_packages.txt
Contains all required packages and libraries

## 1. Problem Statement
- How can bias be identified from political news posts by utilizing convolutional neural networks with natural language processing methods? 
- Efficacy will be evaluated based on performance, error metrics, sentiment analysis and recommendations and guidance will be offered. 
- Interested parties in this research include political party affiliates, media outlets, psychologistics and sociologists, and others intrigued by the phenomenon of media influence upon societal thought processes. 
### We present a novel method for determining bias within abstracts/headlines of online articles, label these articles based on relative bias levels, and use these labels to train a neural network model capable of upscaling this labeling process. Next, we create a custom sentiment analyzer and validate the performance of this analyzer by comparing it to a Valence Aware Dictionary and sEntiment Reasoner (VADER sentiment analyzer) built by a team of professionals. The results of this project will provide the groundwork for future efforts towards understanding bias and sentiment analysis in language processing frameworks. 

## 2. Dataset(s)
- Data will consist of article abstracts and headlines from The New York Times (NYT). 
- These will be imported and converted to `Pandas` DataFrames and edited for content. 
- Cleaning and transformation steps will include utilizing Regular Expressions to make the data conform to standards outlined in greater detail within the project notebook.  
- The results of this analysis will serve as input for basic `NLP` models prior to serving as input for more advanced neural network models.

## 3. Modeling

Part I.
- First, a binary classification model will identify whether or not the abstracts and articles came from an `Opinion` piece or other (1 for `Opinion` and 0 for `other`, all other section names).

- Input features for classification model: Abstract and headline texts.
- Output features for classification model: (1 for `Opinion` and 0 for `other`, all other section names).
  
  Part II. 
  
- Bias scores typical of these modeling projects will be utilized to evaluate the abstracts and headlines for the `Opinion` articles (Cohen Kappa Score generated with Snorkel labeling functions). Please see documents for more detailed explanation. 
  
  Part III.
- Neural Network portion. Model with Long Short Term Memory (LSTM) categorical model, derive sentiment analyzer and compare it with professional VADER sentiment analyzer for performance. 

## 4. Noteworthy Observations
- The benefits of `NLP` models are that they are not computationally expensive compared with their `NN` counterparts. The drawbacks are that they might not contain as many insights as the latter more complex models.
- The benefits of `NN` models are that they have more hyperparameters and customizability compared with their `NLP` counterparts. The drawbacks are that they may be limited by resources such as hardware and computation times necessary to run optimization steps exhaustively for optimal performance, as was the case for this particular study. 
- Assumptions to confirm / requisite steps to take with independent variables for models:
  - Text data must be cleaned, parsed, and transformed into vectors.

## 5. Resources / References
- `API` Documentation from `NYT`: https://developer.nytimes.com/apis
- Code for webscraping adapted from fellow coursemate's group project, with their permission this code was included in the pipeline for this project.
- Snorkel documentation for Cohen Kappa Score labeling: https://www.snorkel.org
- Pew Center Article on Bias: https://www.pewresearch.org/internet/2017/10/19/the-future-of-truth-and-misinformation-online
- VADER Documentation: https://vadersentiment.readthedocs.io
- Kaggle GPU Documentation: https://www.kaggle.com/code/dansbecker/running-kaggle-kernels-with-a-gpu
- CBOW/Skip Gram: https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa
- CBOW/SKip Gram: https://medium.com/@dube.aditya8/word2vec-skip-gram-cbow-b5e802b00390
- KDE Plot Documentation: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
-
