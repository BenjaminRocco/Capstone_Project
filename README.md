# Capstone Project
General Assembly Capstone Project

1. Problem Statement
- How can bias be identified from political news posts by utilizing convolutional neural networks with natural language processing methods? 
- Efficacy will be evaluated based on performance, error metrics, sentiment analysis and recommendations and guidance will be offered. 
- Interested parties in this research include political party affiliates, media outlets, psychologistics and sociologists, and others intrigued by the phenomenon of media influence upon societal thought processes. 

2. Dataset(s)
- Data will consist of media headlines from `NYT`.
- These will be imported and converted to `Pandas` DataFrames and edited for content. 
- Cleaning and transformation steps will include utilizing Regular Expressions to make the data confirm to standards outlined in greater detail within the project notebook, as well as censoring foul language from strings. 
- The results of this analysis will serve as input for basic `NLP` models prior to serving as input to more advanced neural network models.

3. Modeling

- First, a binary classification model will identify whether or not the article came from an `Opinion` piece or other (1 for `Opinion` and 0 for `other`, all other section names).

- Input features for classification model: Abstract and headline contents.
- Output features for classification model: (1 for `Opinion` and 0 for `other`, all other section names).
- Next, once this model is adequately hyperparameter-tuned sentiment analysis will be conducted on the actual `Opinion` articles (their abstracts and headlines).
  
- Bias scores typical of these modeling projects will be the target variables (multi-classification, as there will be varying scales for each of these two variables).
- `NLP` classification models.
  - Unsupervised
- `NN` classification models.
  -  Supervised
- The author anticipates `NN` will perform exceptionally well compared with the basic `NLP` models. 
- The benefits of `NLP` models are that they are not computationally expensive compared with their `NN` counterparts. The drawbacks are that they might not contain as many insights as the latter more complex models.
- The benefits of `NN` models are that they have more hyperparameters and customizability compared with their `NLP` counterparts. The drawbacks are that they may be limited by resources such as hardware and computation times necessary to run optimization steps exhaustively for optimal performance. 
- Assumptions to confirm / requisite steps to take with independent variables for models:
  - Text data must be cleaned, parsed, and transformed into vectors.

4. Resources / References
- `API` Documentation from `NYT`
