'''Export caffe model config and layer params to disk.'''

from __future__ import print_function

import os
os.environ['GLOG_minloglevel'] = '2'  # Hide caffe debug info.

import json
import caffe
import numpy as np

from prototxt_parser import PrototxtParser


PARAM_DIR = './param/'    # Directory for saving layer param.
CONFIG_DIR = './config/'  # Directory for saving network config.


def dump_param(net, layer_name):
    '''Dump layer params to disk.
    For CONV, LINEAR, SCALE layer, save weight & bias.
    For BN, save running_mean & running_var.

    Save weight & running_mean as '*.w.npy'.
    Save bias & running_var as '*.b.npy'.
    '''
    layer = net.params.get(layer_name)
    if not layer:  # For layer has no params, return.
        return

    # Save weight.
    weight = net.params[layer_name][0].data
    np.save(PARAM_DIR + layer_name + '.w', weight)

    # Save bias, if exists.
    # Note for BatchNorm layer, the attribute `use_global_stats`,
    # which is stored in `net.params[layer_name][2]`, is ommited here.
    if len(layer) > 1:
        bias = net.params[layer_name][1].data
        np.save(PARAM_DIR + layer_name + '.b', bias)


if __name__ == '__main__':
    if not os.path.isdir(PARAM_DIR):
        os.mkdir(PARAM_DIR)

    if not os.path.isdir(CONFIG_DIR):
        os.mkdir(CONFIG_DIR)

    prototxt = './model/t.prototxt'
    binary = './model/t.caffemodel'

    net = caffe.Net(prototxt, binary, caffe.TEST)
    parser = PrototxtParser(prototxt)

    # Parse model layer by layer.
    print('\n==> Exporting layers..')
    SUPPORTED_LAYERS = ['Data', 'DummyData', 'Convolution',       \
                        'BatchNorm', 'Scale', 'ReLU', 'Pooling',  \
                        'Flatten', 'InnerProduct', 'Dropout', 'Softmax']

    for i, layer in enumerate(parser.layers):
        print('... Layer %d : %s' % (i, layer['type']))

        if layer['type'] not in SUPPORTED_LAYERS:
            raise TypeError('%s layer not supported yet!' % layer['type'])

        # Dump layer params to disk (if it has).
        dump_param(net, layer['name'])
