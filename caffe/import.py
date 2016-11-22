'''Import layer params from disk to rebuild the caffemodel.'''

from __future__ import print_function

import os
os.environ['GLOG_minloglevel'] = '2' # Hide caffe debug info.

import json
import caffe
import numpy as np

from caffe import layers as L


# Directory containing layer params and config file.
PARAM_DIR = './param/'
CONFIG_DIR = './config/'


def input_layer(layer_config):
    input_shape = layer_config['input_shape']
    return L.DummyData(shape=[dict(dim=input_shape)], ntop=1)

def conv_layer(layer_config, bottom_name):
    num_output = layer_config['num_output']
    kW, kH = layer_config['kW'], layer_config['kH']
    dW, dH = layer_config['dW'], layer_config['dH']
    pW, pH = layer_config['pW'], layer_config['pH']

    return L.Convolution(num_output=num_output,
                         bottom=bottom_name,
                         kernel_w=kW,
                         kernel_h=kH,
                         stride_w=dW,
                         stride_h=dH,
                         pad_w=pW,
                         pad_h=pH)

def bn_layer(layer_config, bottom_name):
    return L.BatchNorm(bottom=bottom_name,
                       use_global_stats=True)

def scale_layer(layer_config, bottom_name):
    return L.Scale(bottom=bottom_name,
                   bias_term=True)

def relu_layer(layer_config, bottom_name):
    return L.ReLU(bottom=bottom_name)

def pool_layer(layer_config, bottom_name):
    pool_type = layer_config['pool_type']
    kW, kH = layer_config['kW'], layer_config['kH']
    dW, dH = layer_config['dW'], layer_config['dH']
    pW, pH = layer_config['pW'], layer_config['pH']

    return L.Pooling(bottom=bottom_name,
                     pool=pool_type,
                     kernel_w=kW,
                     kernel_h=kH,
                     stride_w=dW,
                     stride_h=dH,
                     pad_w=pW,
                     pad_h=pH)

def flatten_layer(layer_config, bottom_name):
    return L.Flatten(bottom=bottom_name)

def linear_layer(layer_config, bottom_name):
    num_output = layer_config['num_output']
    return L.InnerProduct(bottom=bottom_name,
                          num_output=num_output)

def softmax_layer(layer_config, bottom_name):
    return L.Softmax(bottom=bottom_name)

def build_prototxt():
    '''Build a new prototxt from config file.

    Save as 'cvt_net.prototxt'.
    '''
    print('==> Building prototxt..')

    # Map layer_type to its processing function.
    layer_fn = {
        'Data': input_layer,
        'DummyData': input_layer,
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

    # Build other layers based on config file.
    with open(CONFIG_DIR + 'net.json', 'r') as f:
        net_config = json.load(f)

    # Add input layer.
    print('... Add layer: DummyData')
    input_layer_name = net_config[0]['name']
    net[input_layer_name] = input_layer(net_config[0])

    # DFS graph to build prototxt.
    graph = np.load(CONFIG_DIR + 'graph.npy')
    num_nodes = graph.shape[0]
    marked = [False for i in range(num_nodes)]

    def dfs(G, v):
        marked[v] = True
        bottom_layer_name = net_config[v]['name']
        for w in range(num_nodes):
            if G[v][w] == 1 and not marked[w]:
                layer_config = net_config[w]
                layer_name = layer_config['name']
                layer_type = layer_config['type']

                print('... Add layer: %s' % layer_type)
                get_layer = layer_fn.get(layer_type)
                if not get_layer:
                    raise TypeError(layer_type + ' not supported yet!')

                layer = get_layer(layer_config, bottom_layer_name)
                net[layer_name] = layer
                dfs(G, w)

    # DFS.
    dfs(graph, 0)

    # Save prototxt.
    with open('cvt_net.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
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
    for i in range(len(net.layers)):
        layer_name = net._layer_names[i]
        layer_type = net.layers[i].type

        print('... Layer %d : %s' % (i, layer_type))

        weight, bias = load_params(layer_name)

        if weight is not None:
            net.params[layer_name][0].data[...] = weight
        if bias is not None:
            net.params[layer_name][1].data[...] = bias

        if layer_type == 'BatchNorm':
            net.params[layer_name][2].data[...] = 1.  # use_global_stats=true

    net.save('cvt_net.caffemodel')
    print('Saved!')


if __name__ == '__main__':
    # Build new prototxt based on config file.
    build_prototxt()

    # Fill network with saved params.
    fill_params()
