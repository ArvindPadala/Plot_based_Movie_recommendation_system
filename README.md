# Plot_based_Movie_recommendation_system:
Movie Recommendation System Using Transformer Models
This project presents a plot-driven movie recommendation system leveraging Natural Language Processing (NLP) and Transformer models. Unlike traditional systems that rely on user ratings or viewing history, this system focuses on movie plot summaries and metadata to deliver unbiased, narrative-based recommendations.

# Key Features:

## Genre Classification:
Predicts movie genres from user queries using BERT.
Achieved ~61% accuracy, outperforming baseline models like Naive Bayes.

## Semantic Matching:
Uses Sentence-BERT embeddings for plot similarity matching.
Recommends movies based on cosine similarity of plot embeddings.

## Interactive Question Answering:
Fine-tuned DistilBERT extracts metadata details (e.g., lead actor, director).

# Workflow:
1. User Input: Text query describing movie preferences.
2. Genre Prediction: Predict genres using fine-tuned BERT.
3. Recommendation: Retrieve top similar movies using semantic matching.
4. Q&A: Respond to specific user queries about the recommended movies.

# Technologies Used
Transformers: BERT, DistilBERT, Sentence-BERT.
Dataset: CMU Movie Summary Corpus.
Frameworks: Hugging Face, PyTorch, Scikit-learn.

# Results
Baseline Accuracy: ~20%.
Transformer Model Accuracy: ~61%.

Improved recommendations for rare genres and nuanced queries.

This project showcases the transformative potential of NLP and Transformer models in building smarter, unbiased recommendation systems. 🚀



