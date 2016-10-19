#---------------------------------------------------------------
# Decompose layer params of a caffemodel to disk.
#---------------------------------------------------------------
from __future__ import print_function

import os
import caffe
import numpy as np
from prototxt_parser import PrototxtParser


def save_param(net, layer_name):
    '''Save layer params to disk.

    For layer:
    - CONV, LINEAR, SCALE: save weight & bias.
    - BN: save running_mean & running_var.

    Saving as:
    - weight/running_mean: as '.w.npy'.
    - bias/running_var: as '.b.npy'.
    '''
    weight = net.params[layer_name][0].data # for bn, weight is running_mean
    bias = net.params[layer_name][1].data   # for bn, bias is running_var
    np.save('./params/'+layer_name+'.w', weight)
    np.save('./params/'+layer_name+'.b', bias)

def logging(file, L):
    '''Write list content to log.'''
    L = [str(x) for x in L]
    file.write('\t'.join(L)+'\n')

def println(L):
    '''Print list content to out'''
    L = [str(x) for x in L]
    print(' '.join(L))

if __name__ == '__main__':
    # 1. define .prototxt parser
    parser = PrototxtParser('./model/net.prototxt')

    # 2. load caffe model
    net = caffe.Net('./model/net.prototxt', './model/net.caffemodel', caffe.TEST)

    # mkdir for saving layer params and log file
    save_dir = './params/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    logfile = open(save_dir+'net.log', 'w')

    # 3. parse params layer by layer
    print('\nexporting..')
    for i in range(1, len(net.layers)):  # skip the Input layer (i=0)
        layer_type = net.layers[i].type
        layer_name = net._layer_names[i]
        layer_config = []                # layer configs for logging

        # TODO: Add softmax
        if layer_type not in ['Input', 'InnerProduct', 'Convolution', 'BatchNorm', \
                                'Scale', 'ReLU', 'Pooling', 'Flatten']:
            raise Exception(layer_type+' layer not supported yet!')

        # save layers params
        if layer_type in ['InnerProduct', 'Convolution', 'BatchNorm', 'Scale']:
            save_param(net, layer_name)

        # get layer config from prototxt
        if layer_type in ['Convolution', 'Pooling']:
            layer_config = parser.get_config(layer_name)

        # printing
        print('==> layer', i, ':', layer_type)

        # logging
        info = [i, layer_type, layer_name]
        logging(logfile, info + layer_config)

    logfile.close()
