#---------------------------------------------------------------
# Decompose layer params of a caffemodel to disk.
#---------------------------------------------------------------
from __future__ import print_function

import os
import caffe
import numpy as np
from prototxt_parser import PrototxtParser


def save_linear(net, layer_name, save_dir):
    '''Save linear layer params to disk.

    Returns:
        input_size: input dim of linear layer
        output_size: output dim of linear layer
    '''
    weight = net.params[layer_name][0].data
    bias = net.params[layer_name][1].data
    input_size = weight.shape[1]
    output_size = weight.shape[0]
    # save params
    np.save(save_dir+layer_name+'_weight', weight)
    np.save(save_dir+layer_name+'_bias', bias)
    return input_size, output_size

def save_conv(net, layer_name, save_dir):
    '''Save conv layer params to disk.'''
    weight = net.params[layer_name][0].data
    bias = net.params[layer_name][1].data
    np.save(save_dir+layer_name+'_weight', weight)
    np.save(save_dir+layer_name+'_bias', bias)

def save_bn(net, layer_name, save_dir):
    '''Save bn layer params to disk.'''
    running_mean = net.params[layer_name][0].data
    running_var = net.params[layer_name][1].data
    # momentum = net.params[layer_name][2].data
    np.save(save_dir+layer_name+'_mean', running_mean)
    np.save(save_dir+layer_name+'_var', running_var)

def logging(file, L):
    '''Write list content to log.'''
    s = ''
    for x in L:
        s = s + str(x) + '\t'
    file.write(s+'\n')

if __name__ == '__main__':
    # 1. define .prototxt parser
    parser = PrototxtParser('./model/net.t7.prototxt')

    # 2. load caffe model
    net = caffe.Net('./model/net.t7.prototxt', './model/net.t7.caffemodel', caffe.TEST)

    # mkdir for saving layer params and log file
    save_dir = './params/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    logfile = open(save_dir+'net.log', 'w')

    # 3. parse params layer by layer
    print('\nexporting..')
    for i in range(len(net.layers)):
        layer_type = net.layers[i].type
        layer_name = net._layer_names[i]

        if layer_type == 'InnerProduct':
            # save
            input_size, output_size = save_linear(net, layer_name, save_dir)
            # logging
            print('==> layer', i, ': Linear [', str(input_size), '->', str(output_size), ']')
            logging(logfile, [i, 'Linear', layer_name])
        elif layer_type == 'ReLU':
            # logging
            print('==> layer', str(i), ': ReLU')
            logging(logfile, [i, 'ReLU', layer_name])
        elif layer_type == 'Flatten':
            # logging
            print('==> layer', i, ': Flatten')
            logging(logfile, [i, 'Flatten', layer_name])
        elif layer_type == 'Convolution':
            # save
            save_conv(net, layer_name, save_dir)
            # logging
            kW,kH,dW,dH,pW,pH = parser.get_params(layer_name)
            print('==> layer', i, ': Convolution [', kW,kH,dW,dH,pW,pH, ']')
            logging(logfile, [i, 'Convolution', layer_name, kW,kH,dW,dH,pW,pH])
        elif layer_type == 'BatchNorm':
            # save
            save_bn(net, layer_name, save_dir)
            # logging
            print('==> layer', i, ': BatchNorm')
            logging(logfile, [i, 'BatchNorm', layer_name])
        elif layer_type == 'Pooling':
            # logging
            pool_type,kW,kH,dW,dH,pW,pH = parser.get_params(layer_name)
            print('==> layer', i, ': Pooling')
            logging(logfile, [i, 'Pooling', layer_name,pool_type,kW,kH,dW,dH,pW,pH])


    logfile.close()



#
# net._layer_names[2]
# net.layers[2].type
#
# name = net._layer_names[2]
#
# running_mean = net.params[name][0].data
# running_var = net.params[name][1].data
# momentum = net.params[name][2].data
