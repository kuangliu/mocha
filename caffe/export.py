'''Export caffe model layer params to disk.'''

from __future__ import print_function

import os
os.environ['GLOG_minloglevel'] = '2'  # Hide caffe debug info.
import sys
import json

import caffe

import numpy as np

from prototxt_parser import PrototxtParser


PARAM_DIR = './param/'   # Directory for saving layer params.
CONFIG_DIR = './config/'  # Directory for saving network configs.


def dump_param(net, layer_name):
    '''Save layer params to disk.

    For CONV, LINEAR, SCALE, save weight & bias.
    For BN, save running_mean & running_var.

    Save weight & running_mean as '*.w.npy'.
    Save bias & running_var as '*.b.npy'.
    '''
    layer = net.params.get(layer_name)
    if not layer:   # For layer has no params, return.
        return

    # Save weight.
    weight = net.params[layer_name][0].data
    np.save(PARAM_DIR + layer_name + '.w', weight)

    # Save bias, if exists.
    if len(layer) > 1:
        bias = net.params[layer_name][1].data
        np.save(PARAM_DIR + layer_name + '.b', bias)

if __name__ == '__main__':
    # mkdir for saving layer params and configs.
    if not os.path.isdir(PARAM_DIR):
        os.mkdir(PARAM_DIR)

    if not os.path.isdir(CONFIG_DIR):
        os.mkdir(CONFIG_DIR)

    # prototxt = './model/net.prototxt'
    # binary = './model/net.caffemodel'
    prototxt = './cvt_net.prototxt'
    binary = './cvt_net.caffemodel'

    net = caffe.Net(prototxt, binary, caffe.TEST)
    parser = PrototxtParser(prototxt)

    # Graph representing net structure using adjacent matrix.
    num_layers = len(net.layers)
    graph = np.zeros((num_layers,num_layers))

    # Parse model params layer by layer.
    print('\n==> Exporting layers..')
    SUPPORTED_LAYERS = ['Input', 'Data', 'DummyData', 'Convolution',  \
                        'BatchNorm', 'Scale', 'ReLU', 'Pooling',      \
                        'Flatten', 'InnerProduct', 'Dropout', 'Softmax']

    net_config = []
    for i in range(num_layers):
        layer_type = net.layers[i].type
        layer_name = net._layer_names[i]

        # Use 'DummyData' layer instead of 'Input' layer.
        if layer_type == 'Input':
            layer_type = 'DummyData'
            layer_name = parser.input_layer_name

        print('... Layer %d : %s' % (i, layer_type))

        if layer_type not in SUPPORTED_LAYERS:
            raise TypeError(layer_type + ' layer not supported yet!')

        # Dump layer params to disk (if it has).
        dump_param(net, layer_name)

        # Get layer_config.
        layer_config = {'id'  : i,
                        'type': layer_type,
                        'name': layer_name}
        layer_config.update(parser.get_layer_config(layer_name))

        # Add layer_config to net_config.
        net_config.append(layer_config)

        # Add node to graph.
        # TODO: build graph based on prototxt.
        graph[i][i] = 1
        if i < num_layers - 1:
            graph[i][i+1] = 1

    # Dump layer config to file.
    with open(CONFIG_DIR + 'net.json', 'w') as f:
        json.dump(net_config, f, indent=2)

    # Saving graph adj matrix to file.
    np.save(CONFIG_DIR + 'graph.npy', graph)
