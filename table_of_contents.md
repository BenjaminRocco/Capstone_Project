TABLE OF CONTENTS

├── Data - All generated files from notebooks
│   ├── df_with_section_names.csv
│   ├── dfnew_with_section_names.csv
│   ├── kappa_labeled_train.csv
│   ├── new_labels_v2.csv
│   ├── new_train_neg_words.csv
│   ├── precleaned_df.csv
│   ├── sampled_test_hand_labeled.csv
│   ├── sampled_test_to_hand_label.csv
│   ├── sampled_train.csv
│   ├── section_marked_train.csv
│   ├── united_states.parquet
│   ├── united_states_headlines_last_decade.csv
│   ├── unlabeled_test.csv
│   └── unlabeled_train.csv

├── Documentation
│   ├── Classification_Model_Basis.md
│   ├── Label_Function_Revision_Process.md
│   ├── Label_Functions_Explanation.md
│   ├── Label_Voters_Analogy.md
│   ├── Labeling_Functions_Technical.md
│   ├── Sentiment_Analysis_Portion.md
│   ├── Summary_Findings.md
│   └── Vader_VersusPredModel_SentimentTest_1.rtf

├── README.md

├── Visualizations - For inclusion within Powerpoint Presentation
│   ├── CBOW_QUADS.png
│   ├── CBOW_TOTALITY.png
│   ├── KDEPlots_NN.png
│   ├── KDEPlots_SampleSizeSmall.png
│   ├── KDE_Macro_NN.png
│   ├── KDE_Macro_Sent.png
│   ├── SVC_TVEC_CONMATRIX.png
│   ├── VADER.png
│   ├── abhead_couplings_grouped_basedon_section.png
│   ├── biased_words.png
│   ├── mad_fracas_allwords.png
│   ├── skipgram_architecture.png
│   ├── snorkel_square.png
│   ├── textblob.jpeg
│   ├── unbiased_words.png
│   ├── words_similar_context_tSNE.png
│   └── words_similar_word2vec.png

├── binary_classification_SVC_nongrid.pkl - Pickled Support Vector Machine model for binary classification.
├── binary_classification_TVEC_nongrid.pkl - Pickled TfidfVectorizer for binary classification. 

├── model_11_serial - Keras Long Short Term Memory Neural Network Model. Must be loaded in this format locally and within deployed Streamlit App. 
│   ├── assets
│   ├── fingerprint.pb
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index

├── part_01
│   ├── Code
│   │   ├── CleaningEDA_ClassModeling.ipynb - Binary Classification models to identify Opinion versus non-Opinion section text strings.
│   │   ├── Data_Collection_NYT.ipynb - Webscraping code from NYT. Placeholder credentials, user must create own profile and fill with their credentials. 
│   │   ├── binary_classification_SVC_nongrid.pkl - Outputs from 
│   │   ├── binary_classification_TVEC_nongrid.pkl
│   │   └── utilities.py
│   └── __pycache__
│       └── utilities.cpython-310.pyc

├── part_02
│   ├── Labeling_Funcs_Snorkel.ipynb - Snorkel Functions and label analysis. 
│   └── Part2_Data_Labeling_Weighted_Function.ipynb - Notebook to create Linear labeling function. 

├── part_03
│   ├── Part3_LSTM_nn_kappa.ipynb - Notebook to create LSTM categorical neural network.
│   ├── Word_Cloud_Visuals.ipynb - Notebook for generating Word Cloud visualizations of handlabeled Snorkel data. 
│   ├── cbow_weights.h5 - CBOW weights file. 
│   ├── model_structure_skip.png - Flowchart of Skip Grams neural network, nearly identical to CBOW. 
│   ├── statistical_analysis.ipynb - See for technical comparisons between neural network model and labeling function results as well as BEAST and VADER analyzers.
│   ├── word2vec-skipgram.ipynb -Skip Grams notebook
│   └── word2vec_cbow.ipynb - CBOW notebook

├── required_libraries_packages.txt
├── requirements.txt

├── streamlit_script.py - Deployment version
├── streamlit_script_local.py - Local version only
├── table_of_contents.md
└── Capstone_Presentation.pdf

13 directories, 77 files
