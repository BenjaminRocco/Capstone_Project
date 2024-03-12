# Revising Our Labeling Function and Fitting to `VADER` Sentiment Analyzer

How did we continue to develop our labeling functions and fit our sentiment analyzer to the `VADER` sentiment analyzer? Well, apart from analyzing, using Snorkel’s built-in metric functionality to evaluate how label functions were performing on hand-labeled validation set data (documentation in `Labeling_Functions_Technical.md`), we utilized `Word2Vec`, `Continuous-Bag-Of-Words`, and `Skip Gram`, three similar but different word encoders to gain valuable insights into the textual information we gathered for analysis. 

# Utilization of `Word2Vec`, A Gentle Introduction! 

What is `Word2Vec`? Essentially, it is a way of prescribing vector like qualities/quantities to a word within a group of words. What are those fundamental qualities/quantities? Namely, qualitatively they are a magnitude (length) and direction. Quantitatively, the length has a scalar (constant) value while the direction has components pointing to directions spanning the space! Think: $x$ and $y$ in $\textbf{2D}$ space (cartesian); $x,y,z$ in $\textbf{3D}$ space (cartesian), and different values in various coordinate systems and non-cartesian spaces beyond the scope of this project! Beyond the theory and in practice, `Word2Vec` is a popular algorithm that generates word embeddings that can represent words as dense vectors in continuous vector space. These embeddings clearly define semantic and syntactic relationships between words and are used in various natural language processing tasks. There are also algorithms very similar to `Word2Vec` that serve different purposes. For our project, we utilized these algorithms in order to develop meaningful relationships amongst words that could potentially lead us to forming insightful conclusions regarding biased language. These functions enabled us to change our approach to defining labeling functions in order to decrease the error in performance of our sentiment analyzer and `VADER`’s sentiment analyzer. In the following passage we detail exactly how we were able to accomplish these feats. 

### Continuous-Bag-Of-Words (CBOW)

There is a process to all of these algorithms, and the process is as follows: 

1. Utilize Regular Expressions (Regex) to preprocess words and phrases. We refer to these things collectively as a text corpus, from the latin work corpus meaning body. This always involves tokenizing text into individual words, removing punctuation and other characters, and converting all text into lowercase. 

2. In Continuous-Bag-Of-Words (CBOW), our objective is to predict a target word given we have the surrounding words within a specific radius of words from our target. We form training samples by roving over the text corpus in segmented lengths and forming (context, target) pairs. Think of it as a partition over our text phrases, the union of which is the entire corpus! 

3. Next comes the encoding step. We convert our context words and target words into one-hot-encoded vectors. This means that each vector will have the size of its vocabulary, with all zeros aside from a one at the index corresponding to the word’s position in the vocabulary. Think of these as unit vectors pointing in a unique direction in our space! If we had only two words, we would have $[0,1],\,[1,0]$, three: we would have $[1, 0, 0],\,[0, 1, 0],\,[0,0,1]$, etc.

4. Now we build our model! We created a neural network model with an embedding layer, hidden layers, and an output layer. The embedding layer mapped the one-hot-encoded context words to dense vectors and the hidden layer performed transformations, where the output layer predicted the target word. 

5. Model Training step. We input our (context, target) pairs and minimize error, i.e. the difference between our predicted target words and actual target words. For our purposes we utilize an Adam optimizer. 

6. Finally, we extract the word embeddings and examine the weights of the embedding layer either with graphs, which show words that are closer in meaning and context being closer together, and words more dissimilar in meaning being further away. 

We can also utilize cosine similarity, which means looking at the angle between these formed vectors in order to tell how similar they are, since we did in fact assign them directions! If we recall from geometry, two lines that overlap going in the same direction have a cosine(0 degrees) and value of $1.$ Well, this means that two words that are exactly the same will have cosine similarity of $1$! Their angle between one another is literally $0.$ Words that have nothing to do with one another point in the unit vector directions, they have components [1,0] and [0,1] for example; the angle between them is 90 degrees, and thus, cosine(90)=0 . These dissimilar words will have a cosine similarity value of zero! Now that we have explained how we can utilize known mathematical concepts as a “sanity check” for our CBOW algorithm, we move on to Skip Gram and how it performs in a different way than CBOW. 

# Skip Gram

1. Utilize Regular Expressions (Regex) to preprocess words and phrases. This always involves tokenizing text into individual words, removing punctuation and other characters, and converting all text into lowercase, in the exact same way as CBOW. 

2. Context and Target word creation is the exact opposite of CBOW! Given a target word, our objective becomes predicting the words surrounding it, within a certain word-radius. As with CBOW, we form training samples by roving over the text corpus in segmented lengths and forming (context, target) pairs.

3. Next comes the encoding step, which is identical to CBOW. 

4. Now we build our model! It has a similar architecture as the CBOW model. 

5. Model Training step. Surprise! It is also similar to the CBOW training protocol! 

6. Finally, we extract the word embeddings and examine the weights of the embedding layer either with graphs, which show words that are closer in meaning and context being closer together, and words more dissimilar in meaning being further away. 

We may utilize principal component analysis to reduce the dimensionality of these clusters. As can be observed from our graphs, we have high dimensionality due to the large quantity of data (25,000 samples) we have amassed. Therefore, we reduce the dimensionality of our clusters with a method known as principal component analysis (PCA) in order to gain more insight into our clusters. 

### In Conclusion

The insights we gained from these methods helped further develop label functions in order to close the gap between our sentiment analyzer and VADER’s sentiment analyzer. By gaining valuable information regarding contexts of words contained in passages with greater tendency for bias, we were able to target those words for labeling functions, effectively label abstract/headline pairs perceived as tending towards more bias in nature, and close the margin of error between our homebrewed sentiment analyzer and the professionally designed VADER sentiment analyzer. 












