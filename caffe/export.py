#---------------------------------------------------------------
# Export layer params of a caffemodel to disk.
#---------------------------------------------------------------

from __future__ import print_function

import os
os.environ['GLOG_minloglevel'] = '2' # hide caffe debug info

import sys
import numpy as np
import caffe

from prototxt_parser import PrototxtParser


def save_param(net, layer_name):
    '''Save layer params to disk.

    For layer:
      - CONV, LINEAR, SCALE: save weight & bias (optional).
      - BN: save running_mean & running_var.

    Saving as:
      - weight/running_mean: as '.w.npy'.
      - bias/running_var: as '.b.npy'.
    '''
    N = len(net.params[layer_name])
    assert N > 0, 'No param in layer ' + layer_name
    # save weight
    weight = net.params[layer_name][0].data
    np.save(save_dir + layer_name + '.w', weight)
    # save bias (if have)
    if N > 1:
        bias = net.params[layer_name][1].data
        np.save(save_dir + layer_name + '.b', bias)

def logging(file, L):
    '''Write list content to log.'''
    L = [str(x) for x in L]
    file.write('\t'.join(L)+'\n')

def println(L):
    '''Print list content to out'''
    L = [str(x) for x in L]
    print(' '.join(L))

if __name__ == '__main__':
    # mkdir for saving layer params and log file
    save_dir = './params/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    prototxt = './model/net.prototxt'
    binary = './model/net.caffemodel'
    # prototxt = '/home/luke/workspace/child/model/child.prototxt'
    # binary = '/home/luke/workspace/child/model/child.caffemodel'

    # 1. define prototxt parser
    parser = PrototxtParser(prototxt)

    # 2. load caffe model
    net = caffe.Net(prototxt, binary, caffe.TEST)

    # 3. parse params layer by layer
    logfile = open(save_dir+'net.log', 'w')
    print('\n==> exporting..')
    for i in range(1, len(net.layers)):  # skip the Input layer (i=0)
        layer_type = net.layers[i].type
        layer_name = net._layer_names[i]
        layer_config = []                # layer configs for logging

        if layer_type not in ['Input', 'Convolution', 'BatchNorm',      \
                              'Scale', 'ReLU', 'Pooling', 'Flatten',    \
                              'InnerProduct', 'Dropout', 'Softmax']:
            raise TypeError(layer_type + ' layer not supported yet!')

        # save layers params
        if layer_type in ['Convolution', 'BatchNorm', 'Scale', 'InnerProduct']:
            save_param(net, layer_name)

        # get layer config from prototxt
        if layer_type in ['Convolution', 'Pooling', 'Dropout', 'InnerProduct']:
            layer_config = parser.get_config(layer_name)

        # printing
        print('layer', i, ':', layer_type)

        # logging
        info = [i, layer_type, layer_name]
        logging(logfile, info + layer_config)

    logfile.close()
