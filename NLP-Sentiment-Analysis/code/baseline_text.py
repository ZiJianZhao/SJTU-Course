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
from text_io import load_bin_vec

def clean_str( review, remove_stopwords = False):
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops] 
    words = " ".join(words)
    return(words)

def read_file_with_word2vec(filename, word2vec):
    examples = list(open(filename).readlines())
    examples = [s.strip() for s in examples]
    label = [int(sent.split(" ")[0]) for sent in examples]
    sents = [clean_str(sent).split(" ") for sent in examples]
    data = []
    for sent in sents:
        num = 0
        tmp = np.zeros((300,))
        for word in sent:
            if word in word2vec:
                tmp += word2vec[word]
                num += 1        
        if num == 0:
            tmp = np.random.random((300,))
        else:
            tmp = tmp / num
        data.append(tmp)
    data = np.array(data)
    label = np.array(label)
    return [data, label]

def get_vocab(filename):
    examples = list(open(filename).readlines())
    examples = [s.strip() for s in examples]
    label = [int(sent.split(" ")[0]) for sent in examples]
    sents = [clean_str(sent).split(" ") for sent in examples]
    vocab = {}
    for sent in sents:
        for word in sent:
            if word not in vocab:
                vocab[word] = 1
    return vocab

def read_file(filename = "../data/train.txt"):
    examples = list(open(filename).readlines())
    examples = [s.strip() for s in examples]
    label = [int(sent.split(" ")[0]) for sent in examples]
    data = [clean_str(sent) for sent in examples]
    return [data, label]

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print 'Loading google word2vec, this may take a while ...'
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs