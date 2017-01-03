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
from baseline_text import get_vocab, read_file_with_word2vec, load_bin_vec

def clean_str( review, remove_stopwords = False):
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    words = " ".join(words)
    return(words)

def read_file(filename = "../data/train.txt"):
    examples = list(open(filename).readlines())
    examples = [s.strip() for s in examples]
    label = [int(sent.split(" ")[0]) for sent in examples]
    data = [clean_str(sent) for sent in examples]
    return [data, label]

def draw_graph(filename = '../data/train.txt'):

    examples = list(open(filename).readlines())
    examples = [s.strip() for s in examples]
    label = [int(sent.split(" ")[0]) for sent in examples]
    data = [clean_str(sent).split(" ") for sent in examples]
    max_length = max([len(sent) for sent in data])
    lx = range(max_length+1)
    length = [0] * (max_length+1)
    for sent in data:
        length[len(sent)] += 1
    max_num = max(length)
    plt.bar(lx, length, color='black', alpha=1)
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.title('Review Length Count')
    plt.axis([-1, max_length, 0, max_num*1.2])
    plt.grid(True)
    plt.show()

    label = np.array(label)
    total = float(len(label))
    x = range(5)
    y = [0] * 5
    for i in x:
        y[i] = sum(label == i) / total
        x[i] = x[i] - 0.4
    
    plt.bar(x, y, color='black', alpha=1)
    plt.xlabel('label')
    plt.ylabel('% of sentiment')
    plt.title('Label Distribution')
    plt.axis([-1, 5, 0, 1])
    plt.grid(True)
    plt.show()


def draw_t_sne_graph(
    filename = '../data/train.txt',
    word2vec_file = '../GoogleNews-vectors-negative300.bin'
):
    vocab = get_vocab(train_file)
    word2vec = load_bin_vec(word2vec_file, vocab)
    train_data, train_label = read_file_with_word2vec(filename, word2vec)
    data = []
    label = np.zeros((500,))
    num = -1
    for i in range(500):
        if i % 100 == 0:
            num += 1
        label[i] = num
    for i in range(5):
        idx = np.where(train_label == i)[0]
        np.random.shuffle(idx)
        part = idx[0:100].tolist()
        for j in part:
            data.append(train_data[j, :])
    data = np.array(data)
    model = TSNE(n_components=2, random_state=0, learning_rate = 500, n_iter = 2000)
    x = model.fit_transform(data)
    print x[0:2, :]
    colors = ['b', 'm', 'g', 'r', 'y']
    for i in range(5):
        idx = np.where(label == i)[0]
        plt.scatter(x[idx,1], x[idx,0], s=20, marker = 'o', color = colors[i], label='%d' % i)   
    plt.legend(loc = 'upper right')
    plt.title('Word2vec Averaging Visualization')
    plt.show()  

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def draw_confusion(y_test, y_pred, class_names):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, 
        title='Validation Confusion matrix')
    plt.show()



