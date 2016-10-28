'''Import layer params from disk to rebuild the caffemodel.'''
from __future__ import print_function

import os
os.environ['GLOG_minloglevel'] = '2' # hide caffe debug info

import caffe
import numpy as np

from caffe import layers as L


# Directory containing layer params and config file.
PARAM_DIR = './params/'


def conv_layer(layer_name, bottom_name, config):
    num_output = config[0]
    kW, kH = config[1], config[2]
    dW, dH = config[3], config[4]
    pW, pH = config[5], config[6]

    return L.Convolution(num_output=num_output,
                         bottom=bottom_name,
                         kernel_w=kW,
                         kernel_h=kH,
                         stride_w=dW,
                         stride_h=dH,
                         pad_w=pW,
                         pad_h=pH)

def bn_layer(layer_name, bottom_name, config):
    return L.BatchNorm(bottom=bottom_name,
                       use_global_stats=True)

def scale_layer(layer_name, bottom_name, config):
    return L.Scale(bottom=bottom_name,
                   bias_term=True)

def relu_layer(layer_name, bottom_name, config):
    return L.ReLU(bottom=bottom_name)

def pool_layer(layer_name, bottom_name, config):
    pool_type = config[0]
    kW, kH = config[1], config[2]
    dW, dH = config[3], config[4]
    pW, pH = config[5], config[6]

    return L.Pooling(bottom=bottom_name,
                     pool=pool_type,
                     kernel_w=kW,
                     kernel_h=kH,
                     stride_w=dW,
                     stride_h=dH,
                     pad_w=pW,
                     pad_h=pH)

def flatten_layer(layer_name, bottom_name, config):
    return L.Flatten(bottom=bottom_name)

def linear_layer(layer_name, bottom_name, config):
    num_output = config[0]
    return L.InnerProduct(bottom=bottom_name,
                          num_output=num_output)

def softmax_layer(layer_name, bottom_name, config):
    return L.Softmax(bottom=bottom_name)

def build_prototxt(input_size):
    '''Build a new prototxt from config file.

    Args:
      input_size: (list) containing 4 numbers indicating network input size.
        e.g. input_size=[1,1,28,28]

    Save as 'cvt_net.prototxt'.
    '''

    print('==> Building prototxt..')

    # Map layer_type to its processing function.
    layer_fn = {
        'Convolution': conv_layer,
        'BatchNorm': bn_layer,
        'Scale': scale_layer,
        'ReLU': relu_layer,
        'Pooling': pool_layer,
        'Flatten': flatten_layer,
        'InnerProduct': linear_layer,
        'Softmax': softmax_layer
    }

    net = caffe.NetSpec()

    # New data layer.
    bottom_name = 'data'
    net['data'] = L.DummyData(shape=[dict(dim=input_size)], ntop=1)

    # Build other layers based on config file.
    config_file = open(PARAM_DIR + 'net.config', 'r')
    for line in config_file.readlines():
        splited = line.strip().split()
        layer_type = splited[1]
        layer_name = splited[2]

        # Add new layer.
        get_layer = layer_fn.get(layer_type)
        if not get_layer:
            raise TypeError(layer_type + ' not supported yet!')

        layer = get_layer(layer_name, bottom_name, [int(x) for x in splited[3:]])
        net[layer_name] = layer

        # Update bottom layer name.
        bottom_name = layer_name

    # Save as prototxt.
    f = open('cvt_net.prototxt', 'w')
    f.write(str(net.to_proto()))
    f.close()
    print('Saved!\n')

def load_params(layer_name):
    '''Load saved layer params.

    Returns:
      (ndarray) weight or running_mean or None.
      (ndarray) bias or running_var or None.
    '''

    weight_path = PARAM_DIR + layer_name + '.w.npy'
    bias_path = PARAM_DIR + layer_name + '.b.npy'
    weight = np.load(weight_path) if os.path.isfile(weight_path) else None
    bias = np.load(bias_path) if os.path.isfile(bias_path) else None

    return weight, bias

def fill_params():
    '''Fill network with saved params.

    Save as 'cvt_net.caffemodel'.
    '''

    print('==> Filling layer params..')

    net = caffe.Net('cvt_net.prototxt', caffe.TEST)
    for i in range(1, len(net.layers)):  # Start from 1, skip the data layer.
        layer_name = net._layer_names[i]
        layer_type = net.layers[i].type

        print('... Layer %d : %s' % (i, layer_type))

        weight, bias = load_params(layer_name)
        if weight is not None:
            net.params[layer_name][0].data[...] = weight
        if bias is not None:
            net.params[layer_name][1].data[...] = bias

    net.save('cvt_net.caffemodel')
    print('Saved!')


if __name__ == '__main__':
    # Build new prototxt based on config file.
    build_prototxt(input_size=[1,1,28,28])

    # Fill network with saved params.
    fill_params()
