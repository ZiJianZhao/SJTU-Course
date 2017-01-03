#!/usr/bin/env python

import os
import re
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.manifold import TSNE
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from baseline_text import load_bin_vec, read_file_with_word2vec, get_vocab, read_file, load_bin_vec
from visual import draw_confusion

def bag_of_words_feature(
    train_file = '../data/train.txt', 
    test_file = '../data/dev.txt'
):
    train_data, train_label = read_file(train_file)
    test_data, test_label = read_file(test_file)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    print "Creating the bag of words...\n"
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer = None,
        preprocessor = None,
        stop_words = None,   
        max_features = 5000
    )  
    train_data_features = vectorizer.fit_transform(train_data)
    train_data = train_data_features.toarray()
    test_data_features = vectorizer.transform(test_data)
    test_data = test_data_features.toarray()
    return train_data, train_label, test_data, test_label

def google_word2vec_feature(
    train_file = '../data/train.txt', 
    test_file = '../data/dev.txt',
    word2vec_file = '../GoogleNews-vectors-negative300.bin'
):
    vocab = get_vocab(train_file)
    word2vec = load_bin_vec(word2vec_file, vocab)
    train_data, train_label = read_file_with_word2vec('../data/train.txt', word2vec)
    test_data, test_label = read_file_with_word2vec('../data/dev.txt', word2vec)
    return train_data, train_label, test_data, test_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Machine Learning for Sentiment Analysis")
    parser.add_argument('--train_file', default = '../data/train.txt', type = str,
        help = 'training file path')
    parser.add_argument('--valid_file', default = '../data/dev.txt', type = str,
        help = 'validation file path')
    parser.add_argument('--w2v_file', default = '../GoogleNews-vectors-negative300.bin', type = str,
        help = 'pre-trained word vectors file path')
    parser.add_argument('--feature', default = 'w2c', type = str,
        help = 'feature: bow(bag of words); w2v(word2vec)')
    parser.add_argument('--classifier', default = 'lr', type = str,
        help = 'classifier: rf(random forest); lr(Logistic Regression); nb(Naive Bayes)')
    args = parser.parse_args()
    print(args)
    train_file = args.train_file
    valid_file = args.valid_file
    w2v_file = args.w2v_file
    feature = args.feature
    classifier = args.classifier
    if feature == 'bow':
        train_data, train_label, test_data, test_label = bag_of_words_feature(
            train_file = train_file, 
            test_file = valid_file
        )
    else:
        train_data, train_label, test_data, test_label = google_word2vec_feature(
        train_file = train_file, 
        test_file = valid_file,
        word2vec_file = w2v_file
    )
    if classifier == 'rf':
        model = RandomForestClassifier(n_estimators = 150)
    elif classifier == 'lr':
        model = LogisticRegression()
    else:
        model = MultinomialNB()
    
    print "Start Training (this may take a while)..."

    model = model.fit( train_data, train_label)
    # Use the random forest to make sentiment label predictions
    train_result = model.predict(train_data)
    train_num_correct = sum(train_result == train_label)
    train_num_total = len(train_result)
    print "train accuracy %.3f" % (float(train_num_correct) / train_num_total)
    test_result = model.predict(test_data)
    test_num_correct = sum(test_result == test_label)
    test_num_total = len(test_result)
    print "test accuracy %.3f" % (float(test_num_correct) / test_num_total)
    np.set_printoptions(precision=2)
    draw_confusion(test_label, test_result, class_names = ['0', '1', '2', '3', '4'])
