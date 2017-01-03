#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import mxnet as mx
import numpy as np
import time
import math
import re
from collections import namedtuple

from text_io import load_bin_vec, get_embed_with_word2vec

Model = namedtuple("Model", ['executor', 'symbol', 'data', 'label'])

def cnn_for_text(
        sentence_size, num_embed, batch_size, vocab_size,
        num_label = 5, filter_list = [3, 4, 5], num_filter = 200,
        dropout = 0., fixed_embed = True
    ):

    input_x = mx.sym.Variable('data') # placeholder for input
    input_y = mx.sym.Variable('label') # placeholder for output

    # embedding layer
    embed_layer = mx.sym.Embedding(
        data = input_x, 
        input_dim = vocab_size, 
        output_dim = num_embed, 
        name = 'embed'
    )
    if fixed_embed:
        embed_layer = mx.sym.BlockGrad(data = embed_layer)

    conv_input = mx.sym.Reshape(
        data = embed_layer, 
        shape = (batch_size, 1, sentence_size, num_embed)
    )

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(
            data = conv_input, 
            kernel = (filter_size, num_embed), 
            num_filter = num_filter,
            name = 'convolution%d' % i
        )
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(
            data=relui, 
            pool_type='max', 
            kernel=(sentence_size - filter_size + 1, 1), 
            stride=(1,1)
        )
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim = 1)
    h_pool = mx.sym.Reshape(
        data = concat, 
        shape = (batch_size, total_filters)
    )

    # dropout layer
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data = h_pool, p = dropout)
    else:
        h_drop = h_pool

    # fully connected
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')
    fc = mx.sym.FullyConnected(
        data = h_drop, 
        weight = cls_weight, 
        bias = cls_bias, 
        num_hidden = num_label
    )
    # softmax output
    sm = mx.sym.SoftmaxOutput(
        data = fc, 
        label = input_y, 
        name = 'softmax'
    )
    return sm


def setup_cnn_model(
    ctx, batch_size, sentence_size, 
    num_embed, vocab_size, num_label, 
    filter_list = [3, 4, 5], num_filter = 100, dropout = 0.5,  
    fixed_embed = False, use_word2vec = True, 
    word2vec_path = None, word2idx = None
):
    # define symbol
    cnn = cnn_for_text(
        sentence_size = sentence_size, 
        num_embed = num_embed, 
        batch_size = batch_size, 
        vocab_size = vocab_size,
        num_label = num_label, 
        filter_list = filter_list, 
        num_filter = num_filter,
        dropout = dropout, 
        fixed_embed = fixed_embed
    )
    print cnn.list_arguments()
    # bind symbol to executor
    input_shapes = {}
    input_shapes['data'] = (batch_size, sentence_size)
    input_shapes['label'] = (batch_size, )

    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, cnn.list_arguments()):
        if name in ['label', 'data']: # input, output
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    executor = cnn.bind(
        ctx = ctx, 
        args = arg_arrays, 
        args_grad = args_grad, 
        grad_req = 'write'
    )
    
    # initialization
    arg_arrays = dict(zip(cnn.list_arguments(), executor.arg_arrays))
    '''for name, arr in arg_arrays.items():
        if name not in input_shapes:
            initializer(name, arr)'''
    for name, arr in arg_arrays.items():
        if name not in input_shapes:
            if re.match('convolution._weight', name):
                weight =  mx.nd.array(np.random.uniform(-0.01, 0.01, arr.shape))
                weight.copyto(arr)
                #mx.init.Uniform(0.01)(name, arr)
            elif re.match('.*bias', name):
                bias = mx.nd.array(np.zeros(arr.shape))
                bias.copyto(arr)
            elif re.match('cls_weight', name):
                #mx.init.Uniform(0.1)(name, arr)
                weight = mx.nd.array(np.zeros(arr.shape))
                weight.copyto(arr)
            elif re.match('embed_weight', name):
                #mx.init.Uniform(0.25)(name, arr)
                weight =  mx.nd.array(np.random.uniform(-0.25, 0.25, arr.shape))
                weight.copyto(arr)                
                                                
    if use_word2vec:
        word2vec = load_bin_vec(word2vec_path, word2idx)
        embed_weight = get_embed_with_word2vec(word2idx, word2vec)
        embed_weight = mx.nd.array(embed_weight)
        print 'embed_weight with word2vec: ', embed_weight.shape, 
        print 'embed_weight inferred: ', arg_arrays['embed_weight'].asnumpy().shape
        embed_weight.copyto(arg_arrays['embed_weight'])

    data = executor.arg_dict['data']
    label = executor.arg_dict['label']

    return Model(
        executor = executor, 
        symbol = cnn, 
        data = data, 
        label = label
    )

'''
cnn = cnn_for_text(
        sentence_size = 100, 
        num_embed = 100, 
        batch_size = 32, 
        vocab_size = 1000,
        num_label = 5, 
        filter_list = [3,4,5], 
        num_filter = 100,
        dropout = 0, 
        fixed_embed = True
)

print cnn.list_arguments()
'''
