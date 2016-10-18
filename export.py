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
    np.save(save_dir+layer_name+'.weight', weight)
    np.save(save_dir+layer_name+'.bias', bias)
    return input_size, output_size

def save_conv(net, layer_name, save_dir):
    '''Save conv layer params to disk.'''
    weight = net.params[layer_name][0].data
    bias = net.params[layer_name][1].data
    np.save(save_dir+layer_name+'.weight', weight)
    np.save(save_dir+layer_name+'.bias', bias)

def save_bn(net, layer_name, save_dir):
    '''Save bn layer params to disk.'''
    running_mean = net.params[layer_name][0].data
    running_var = net.params[layer_name][1].data
    # momentum = net.params[layer_name][2].data
    np.save(save_dir+layer_name+'.mean', running_mean)
    np.save(save_dir+layer_name+'.var', running_var)

def logging(file, L):
    '''Write list content to log.'''
    s = ''
    for x in L:
        s = s + str(x) + '\t'
    file.write(s+'\n')

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
    for i in range(len(net.layers)):
        layer_type = net.layers[i].type
        layer_name = net._layer_names[i]

        if layer_type == 'Input':
            pass
        elif layer_type == 'InnerProduct':
            input_size, output_size = save_linear(net, layer_name, save_dir)
            print('==> layer', i, ': Linear [', str(input_size), '->', str(output_size), ']')
            logging(logfile, [i, 'Linear', layer_name])
        elif layer_type == 'ReLU':
            print('==> layer', str(i), ': ReLU')
            logging(logfile, [i, 'ReLU', layer_name])
        elif layer_type == 'Flatten':
            print('==> layer', i, ': Flatten')
            logging(logfile, [i, 'Flatten', layer_name])
        elif layer_type == 'Convolution':
            save_conv(net, layer_name, save_dir)
            kW,kH,dW,dH,pW,pH = parser.get_params(layer_name)
            print('==> layer', i, ': Convolution [', kW,kH,dW,dH,pW,pH, ']')
            logging(logfile, [i, 'Convolution', layer_name, kW,kH,dW,dH,pW,pH])
        elif layer_type == 'BatchNorm':
            save_bn(net, layer_name, save_dir)
            print('==> layer', i, ': BatchNorm')
            logging(logfile, [i, 'BatchNorm', layer_name])
        elif layer_type == 'Pooling':
            pool_type,kW,kH,dW,dH,pW,pH = parser.get_params(layer_name)
            print('==> layer', i, ': Pooling')
            logging(logfile, [i, 'Pooling', layer_name,pool_type,kW,kH,dW,dH,pW,pH])
        else:
            #TODO: Add softmax
            print('[ERROR]'+layer_type+' not supported yet!')
            assert False

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
