import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs
import math
from collections import namedtuple

from text_io import read_file, pad_sentences, read_dict, get_text_id, build_vocab, read_test_file

from data_iter import SequenceIter, DummyIter

from cnn import cnn_for_text, setup_cnn_model

Model = namedtuple("Model", ['executor', 'symbol', 'data', 'label'])

# ----------------- 0. Setup logging ------------------------------------------ 
log_file = 'Log'
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(message)s', 
                    datefmt = '%m-%d %H:%M:%S %p',  
                    filename = log_file,
                    filemode = 'w')
logger = logging.getLogger()
console = logging.StreamHandler()  
console.setLevel(logging.DEBUG)
logger.addHandler(console)

# ----------------- 1. Process the data  ---------------------------------------
parser = argparse.ArgumentParser(description="Deep Learning for Sentiment Analysis")
parser.add_argument('--train_file', default = '../data/train.txt', type = str,
    help = 'training file path')
parser.add_argument('--valid_file', default = '../data/dev.txt', type = str,
    help = 'validation file path')
parser.add_argument('--test_file', default = '../data/test.txt', type = str,
    help = 'test file path')
parser.add_argument('--mode', default = 'rand', type = str,
    help = 'training mode: rand, static, non-static')
parser.add_argument('--machine', default = 'cpu', type = str,
    help = 'training machine: gpu, cpu')

parser.add_argument('--w2v_file', default = '../GoogleNews-vectors-negative300.bin', type = str,
    help = 'pre-trained word vectors file path')

args = parser.parse_args()
print(args)
train_file = args.train_file
valid_file = args.valid_file
test_file = args.test_file

w2v_file = args.w2v_file
mode = args.mode
machine = args.machine
if machine == 'gpu':
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu()

if mode == 'rand':
    use_word2vec = False
    fixed_embed = False
elif mode == 'static':
    use_word2vec = True
    fixed_embed = True
else:
    use_word2vec = True
    fixed_embed = False

word2vec_path = w2v_file

vocab_file = 'vocab.txt'

sentences, label = read_file(train_file)
sentences_padded = pad_sentences(sentences)
build_vocab(sentences_padded, vocab_file , size = None)
logging.info('total sentences lines: %d' % len(sentences_padded))
word2idx = read_dict(vocab_file)
logging.info('dict length: %d' % len(word2idx))
valid_sentences, valid_label = read_file(valid_file)
valid_sentences_padded = pad_sentences(valid_sentences)

train_data, train_label = get_text_id(sentences_padded, label, word2idx)
valid_data, valid_label = get_text_id(valid_sentences_padded, valid_label, word2idx)

print 'train data shape: ' , train_data.shape
print 'example: ', train_label[0], '\t=>\t', train_data[0]
print 'valid data shape: ', valid_data.shape
print 'example: ', valid_label[0], '\t=>\t', valid_data[0]

# ---------------------- 2. Params Defination ----------------------------------------
batch_size = 50
num_embed = 300
sentence_size = train_data.shape[1]
vocab_size = len(word2idx)
num_label = 5

# training parameters
params_dir = 'params'
params_prefix = 'sent'

# ---------------------- 3. Data Iterator Defination ---------------------
train_iter = SequenceIter(
    data = train_data, 
    label = train_label, 
    batch_size = batch_size
)
valid_iter = SequenceIter(
    data = valid_data, 
    label = valid_label, 
    batch_size = batch_size
)

# ------------------  4. Load paramters if exists ------------------------------
'''
model_args = {}

if os.path.isfile('%s/%s-symbol.json' % (params_dir, params_prefix)):
    filelist = os.listdir(params_dir)
    paramfilelist = []
    for f in filelist:
        if f.startswith('%s-' % params_prefix) and f.endswith('.params'):
            paramfilelist.append( int(re.split(r'[-.]', f)[1]) )
    last_iteration = max(paramfilelist)
    print('laoding pretrained model %s/%s at epoch %d' % (params_dir, params_prefix, last_iteration))
    tmp = mx.model.FeedForward.load('%s/%s' % (params_dir, params_prefix), last_iteration)
    model_args.update({
        'arg_params' : tmp.arg_params,
        'aux_params' : tmp.aux_params,
        'begin_epoch' : tmp.begin_epoch
        })
'''

# -----------------------  5. Train model ------------------------------------

if not os.path.exists(params_dir):
    os.makedirs(params_dir)

def train_model(
    model, train_iter, valid_iter, batch_size, 
    optimizer = 'rmsprop', max_grad_norm = 5.0,
    learning_rate = 0.0005, epochs = 200
):
    # optimizer definition
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate
    opt.rho = 0.95
    opt.eps = 1e-6
    opt.clip_gradient = 3
    opt.wd = 0.000
    updater = mx.optimizer.get_updater(opt)

    # metric definition
    train_metric = mx.metric.Accuracy()
    valid_metric = mx.metric.Accuracy()

    # training
    data = model.data
    label = model.label
    executor = model.executor
    symbol = model.symbol
    max_acc = 0
    max_epoch = 0
    curr_acc = 0
    logging.info('learning rate: %g' % opt.lr)
    for epoch in range(epochs):
        train_iter.reset()
        train_metric.reset()
        valid_metric.reset()

        t = 0
        for batch in train_iter:
            # Copy data to executor input. Note the [:].
            data[:] = batch.data[0]
            label[:] = batch.label[0]
            
            # Forward
            executor.forward(is_train=True)
            '''print executor.outputs[1].asnumpy().shape
            print executor.outputs[2].asnumpy().shape
            print executor.outputs[3].asnumpy().shape
            raw_input()'''
            # Backward
            executor.backward()

            # Update
            
            norm = 0
            for i, pair in enumerate(zip(symbol.list_arguments(), executor.arg_arrays, executor.grad_arrays)):
                name, weight, grad = pair
                if name in ['label', 'data']:
                    continue
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                #print name, l2_norm
                norm += l2_norm * l2_norm
            norm = math.sqrt(norm)
            
            for i, pair in enumerate(zip(symbol.list_arguments(), executor.arg_arrays, executor.grad_arrays)):
                name, weight, grad = pair
                if name in ['label', 'data']:
                    continue
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)
                updater(i, grad, weight)

            # metric update
            train_metric.update(batch.label, executor.outputs)
            t += 1
            if t % 40 == 0:
                logging.info('epoch: %d, iter: %d, accuracy: %.3f' % (epoch, t, float(train_metric.get()[1])))
        
        
        # save checkpoints
        
        if not os.path.isfile('%s/%s-symbol.json' % (params_dir, params_prefix)):
            model.symbol.save('%s/%s-symbol.json' % (params_dir, params_prefix))
        
        mx.model.save_checkpoint(
            prefix = '%s/%s' % (params_dir, params_prefix), 
            epoch = epoch, 
            symbol = model.symbol, 
            arg_params = model.executor.arg_dict, 
            aux_params = model.executor.aux_dict
        )
        

        # evaluate on dev set
        num_correct = 0
        num_total = 0
        for batch in valid_iter:
            data[:] = batch.data[0]
            label[:] = batch.label[0]
            executor.forward(is_train = False)
            #num_correct += sum(label.asnumpy() == np.argmax(executor.outputs[0].asnumpy(), axis=1))
            #num_total += label.asnumpy().shape[0]
            valid_metric.update(batch.label, executor.outputs)
        curr_acc = valid_metric.get()[1]
        #dev_acc = num_correct * 100 / float(num_total)
        logging.info('===================================')
        logging.info('epoch: %d, validation accuracy: %.3f' % (epoch, valid_metric.get()[1]))
        #logging.info('accuracy: %.3f' % dev_acc)  
        logging.info('===================================')
        if curr_acc >= max_acc:
            max_acc = curr_acc
            max_epoch = epoch
        elif (max_acc - curr_acc) > 0.01 :
            _, model_params, _ = mx.model.load_checkpoint(
                prefix = '%s/%s' % (params_dir, params_prefix), 
                epoch = max_epoch
            )
            logging.info('-----------------------------------')
            logging.info('loading epoch %d parameters' % max_epoch)
            for key in model_params:
                model_params[key].copyto(executor.arg_dict[key])
            opt.lr *= 0.5
            logging.info('reset learning rate to %g' % opt.lr)
            logging.info('-----------------------------------')
    logging.info('max validation accuracy %.3f at epoch %d' % (max_acc, max_epoch))  
    return max_epoch



model = setup_cnn_model(
    ctx = ctx, 
    batch_size = batch_size, 
    sentence_size = sentence_size, 
    num_embed = num_embed, 
    vocab_size = vocab_size, 
    num_label = num_label, 
    filter_list = [2,3,5], 
    num_filter = 300, 
    dropout = 0.5,
    fixed_embed = fixed_embed, 
    use_word2vec = use_word2vec, 
    word2vec_path = word2vec_path, 
    word2idx = word2idx
)

max_epoch = train_model(
    model = model, 
    train_iter = train_iter, 
    valid_iter = valid_iter, 
    batch_size = batch_size, 
    optimizer = 'rmsprop', 
    max_grad_norm = 5.0,
    learning_rate = 0.001,
    epochs = 25
)

# best parameters: (2,3,5), 300, acc: 0.457

# -----------------------  5. Test model ------------------------------------
sentences, label = read_test_file(test_file)
sentences_padded = pad_sentences(sentences)
test_data, test_label = get_text_id(sentences_padded, label, word2idx)

batch_size = 1
for i in range(1,64):
    if test_data.shape[0] % i == 0:
        batch_size = i

test_iter = SequenceIter(
    data = test_data, 
    label = test_label, 
    batch_size = batch_size,
    training = False
)

# training parameters
params_dir = 'params'
params_prefix = 'sent'
sentence_size = test_data.shape[1]
fixed_embed = False
use_word2vec = False
model = setup_cnn_model(
    ctx = ctx, 
    batch_size = batch_size, 
    sentence_size = sentence_size, 
    num_embed = num_embed, 
    vocab_size = vocab_size, 
    num_label = num_label, 
    filter_list = [2,3,5], 
    num_filter = 300, 
    dropout = 0.5,
    fixed_embed = fixed_embed, 
    use_word2vec = use_word2vec, 
    word2vec_path = word2vec_path, 
    word2idx = word2idx
)
executor = model.executor

_, model_params, _ = mx.model.load_checkpoint(
    prefix = '%s/%s' % (params_dir, params_prefix), 
    epoch = max_epoch
)
logging.info('-----------------------------------')
logging.info('loading epoch %d parameters' % max_epoch)
for key in model_params:
    if key not in ['data', 'label']:
        model_params[key].copyto(executor.arg_dict[key])
data = model.data
label = model.label
g = open('test_result.txt', 'w')
logging.info('Genearting labels for test file, written into the test_result.txt')
for batch in test_iter:
    data[:] = batch.data[0]
    label[:] = batch.label[0]
    # Forward
    executor.forward(is_train=False)
    result = np.argmax(executor.outputs[0].asnumpy(), axis=1)
    for ix in result:
        g.write(str(ix)+'\n')
