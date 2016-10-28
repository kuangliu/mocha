'''Export caffe model layer params to disk.'''
from __future__ import print_function

import os
import sys
os.environ['GLOG_minloglevel'] = '2' # hide caffe debug info

import caffe

import numpy as np

from prototxt_parser import PrototxtParser


# Directory for saving layer params & configs.
SAVE_DIR = './params/'


def save_param(net, layer_name):
    '''Save layer params to disk.

    For CONV, LINEAR, SCALE, save weight & bias.
    For BN, save running_mean & running_var.

    Save weight & running_mean as '*.w.npy'.
    Save bias & running_var as '*.b.npy'.
    '''

    num_layers = len(net.params[layer_name])
    assert num_layers > 0, 'ERROR: no param in layer ' + layer_name

    # Save weight.
    weight = net.params[layer_name][0].data
    np.save(SAVE_DIR + layer_name + '.w', weight)

    # Save bias, if exists.
    if num_layers > 1:
        bias = net.params[layer_name][1].data
        np.save(SAVE_DIR + layer_name + '.b', bias)

def logging(file, L):
    '''Logging list content to file.'''
    L = [str(x) for x in L]
    file.write('\t'.join(L)+'\n')


if __name__ == '__main__':
    # mkdir for saving layer params and config file.
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    prototxt = './model/net.prototxt'
    binary = './model/net.caffemodel'

    # Define prototxt parser.
    parser = PrototxtParser(prototxt)

    # Load caffe model.
    net = caffe.Net(prototxt, binary, caffe.TEST)

    # Logging layer configs to config file.
    config_file = open(SAVE_DIR + 'net.config', 'w')

    # Parse model params layer by layer.
    print('\n==> Exporting layers..')
    for i in range(1, len(net.layers)):  # Skip the `Input` layer (i=0).
        layer_type = net.layers[i].type
        layer_name = net._layer_names[i]
        layer_config = []                # Layer configs for logging.

        print('... Layer %d : %s' % (i, layer_type))

        if layer_type not in ['Input', 'Convolution', 'BatchNorm',      \
                              'Scale', 'ReLU', 'Pooling', 'Flatten',    \
                              'InnerProduct', 'Dropout', 'Softmax']:
            raise TypeError(layer_type + ' layer not supported yet!')

        # Save layers params.
        if layer_type in ['Convolution', 'BatchNorm', 'Scale', 'InnerProduct']:
            save_param(net, layer_name)

        # Get layer config from prototxt.
        if layer_type in ['Convolution', 'Pooling', 'Dropout', 'InnerProduct']:
            layer_config = parser.get_config(layer_name)

        # Logging.
        info = [i, layer_type, layer_name]
        logging(config_file, info + layer_config)

    config_file.close()
