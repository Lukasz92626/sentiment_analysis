# IMDb Sentiment Analysis for Movie Reviews

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

Sentiment classification of IMDb movie reviews using multiple Machine Learning and Deep Learning approaches including Logistic Regression, Naive Bayes, SVM, LSTM, and BERT.

## Features

- **5 different models**: Logistic Regression, Naive Bayes, SVM, LSTM, and BERT (fine-tuned)
- **Hyperparameter tuning** - RandomizedSearchCV for classic models
- **Cross-validation** - Stratified 5-fold CV for robust evaluation
- **Model persistence** - Save/load models (pickle, Keras, HuggingFace)
- **Comprehensive metrics** - Accuracy, AUC, Classification Report
- **Error analysis** - False positives/negatives, length correlation
- **Visualizations** - Confusion matrices, ROC curves, WordClouds, performance comparison

## Dataset

[IMDb Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) from Stanford University

- **50,000** labeled movie reviews (25k train / 25k test)
- Balanced classes: 50% positive, 50% negative
- Binary classification: Positive / Negative sentiment

## Installation

### Clone the repository

```bash
git clone https://github.com/Lukasz92626/sentiment_analysis.git
cd sentiment_analysis
