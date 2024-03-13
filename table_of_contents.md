# TABLE OF CONTENTS

## Data - All generated files from notebooks
- df_with_section_names.csv
- dfnew_with_section_names.csv
- kappa_labeled_train.csv
- new_labels_v2.csv
- new_train_neg_words.csv
- precleaned_df.csv
- sampled_test_hand_labeled.csv
- sampled_test_to_hand_label.csv
- sampled_train.csv
- section_marked_train.csv
- united_states.parquet
- united_states_headlines_last_decade.csv
- unlabeled_test.csv
- unlabeled_train.csv

## Documentation
- Classification_Model_Basis.md - Explanation of the starting point for `CleaningEDA_ClassModeling.ipynb`
- Label_Function_Revision_Process.md - Technical document on Word2Vec variants in the label function iterative process
- Label_Functions_Explanation.md - Intended for all viewers. Explanation of labeling functions from a general standpoint
- Label_Voters_Analogy.md - Imaginative analogy of labeling functions as voters in a US electoral college system
- Labeling_Functions_Technical.md - Technical explanation of the Snorkel labeling process and evaluation metrics
- Sentiment_Analysis_Portion.md - Intended for all viewers. A brief explanation regarding the role of sentiment analysis within the project. Less technical viewers may be direct to `Summary_Findings.md` file for information regarding results while technical audiences may refer to `statistical_analysis.ipynb` notebook for more rigorous approach. 
- Summary_Findings.md - Intended for all viewers. General audiences may find this particularly satisfying while technical audiences may refer to `statistical_analysis.ipynb` notebook for deeper analysis.
- largescalerun.rtf - Output from `Statistical_Analysis.ipynb` lambda function on large dataset for reproducing results for future iterations. 

## README.md

## Visualizations - For inclusion within Powerpoint Presentation
- CBOW_QUADS.png
- CBOW_TOTALITY.png
- KDEPlots_NN.png
- KDEPlots_SampleSizeSmall.png
- KDE_Macro_NN.png
- KDE_Macro_Sent.png
- SVC_TVEC_CONMATRIX.png
- VADER.png
- abhead_couplings_grouped_basedon_section.png
- biased_words.png
- mad_fracas_allwords.png
- skipgram_architecture.png
- snorkel_square.png
- textblob.jpeg
- unbiased_words.png
- words_similar_context_tSNE.png
- words_similar_word2vec.png

## Binary Classification Models
- binary_classification_SVC_nongrid.pkl - Pickled Support Vector Machine model
- binary_classification_TVEC_nongrid.pkl - Pickled TfidfVectorizer

## Keras Long Short Term Memory Neural Network Model
- model_11_serial
  - assets
  - fingerprint.pb
  - keras_metadata.pb
  - saved_model.pb
  - variables
    - variables.data-00000-of-00001
    - variables.index

## Part 01
- Code
  - CleaningEDA_ClassModeling.ipynb - Binary Classification models for Opinion vs. non-Opinion sections
  - Data_Collection_NYT.ipynb - Web scraping code from NYT (requires user credentials)
  - utilities.py - Web scraping notebook runs with this file
- __pycache__
  - utilities.cpython-310.pyc

## Part 02
- Labeling_Funcs_Snorkel.ipynb - Snorkel Functions and label analysis
- Part2_Data_Labeling_Weighted_Function.ipynb - Notebook to create Linear labeling function

## Part 03
- Part3_LSTM_nn_kappa.ipynb - Notebook to create LSTM categorical neural network
- Word_Cloud_Visuals.ipynb - Notebook for generating Word Cloud visualizations of hand-labeled Snorkel data
- cbow_weights.h5 - CBOW weights file
- model_structure_skip.png - Flowchart of Skip Grams neural network (similar to CBOW)
- statistical_analysis.ipynb - Technical comparisons between neural network model and labeling function results, BEAST, and VADER analyzers. Legacy notebook for viewing purposes only, as there are unresolved version issues with keras and tensorflow embedding layers. Results are at the end of the notebook beyond the long output.
- word2vec-skipgram.ipynb - Skip Grams notebook - Technical usage for revising labeling functions - source attributed within references section
- word2vec_cbow.ipynb - CBOW notebook - Technical usage for revising labeling functions - source attributed within references section

## Text Files
- required_libraries_packages.txt
- requirements.txt

## Streamlit Deployment
- streamlit_script.py - Deployment version
- streamlit_script_local.py - Local version only

## Other Files
- table_of_contents.md
- Capstone_Presentation.pdf
- model_11_72test.h5 - Original model for usage within `statistical_analysis.ipynb` notebook for all analysis. Due to Keras and Tensorflow issues this model is not able to load properly in the current version of this project. In future versions the author hopes this will be resolved. 

