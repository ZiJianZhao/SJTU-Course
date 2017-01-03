import numpy as np
import re
import itertools
from collections import Counter
import logging
import numpy as np

MAX_LENGTH = 52
PAD = "</s>"

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def read_file(filename = "../data/all.txt"):
    """
    read data from file and preprocess
    """
    examples = list(open(filename).readlines())
    examples = [s.strip() for s in examples]
    sents = [clean_str(sent) for sent in examples]
    data = [sent.split(" ")[1:] for sent in sents]
    label = [int(sent.split(" ")[0]) for sent in sents]
    return [data, label]

def read_test_file(filename = '../data/test.txt'):
    examples = list(open(filename).readlines())
    examples = [s.strip() for s in examples]
    sents = [clean_str(sent) for sent in examples]
    data = [sent.split(" ")[1:] for sent in sents]
    label = [0 for sent in sents]
    return [data, label]    

def pad_sentences(sentences, padding_word = PAD):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(MAX_LENGTH, max(len(x) for x in sentences))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences, outfile = "../data/vocab.txt", size = None):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary = [x[0] for x in word_counts.most_common(size)]
    f = open(outfile, 'w')
    for word in vocabulary:
        f.write(word+'\n')
    f.close()

def read_dict(filename = "../data/vocab.txt"):
    word2idx = {'<UNK>' : 0}
    idx = 1
    with open(filename, 'r') as fid:
        for line in fid:
            line = line.strip(' ').strip('\n')
            if len(line) == 0:
                continue
            if word2idx.get(line) == None:
                word2idx[line] = idx
            idx += 1
    return word2idx

def get_text_id(sentences_padded, labels, word2idx):
    data = np.array([[word2idx.get(word) if word2idx.get(word) else word2idx.get('<UNK>')  for word in sentence] for sentence in sentences_padded])
    label = np.array(labels)
    return data, label

def get_embed_with_word2vec(word2idx, word2vec):
    embed_weight = []
    lis = sorted(word2idx.iteritems(), key=lambda d:d[1], reverse = False)
    total = len(word2idx)
    num = 0
    np.random.seed(1)
    for word, idx in lis:
        if word in word2vec:
            embed_weight.append(word2vec[word])
            num += 1
        else:
            vector = np.random.uniform(-0.25, 0.25, 300)
            embed_weight.append(vector)
    print 'total: %d, idx: %d' % (total, num)
    embed_weight = np.array(embed_weight)
    return embed_weight

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