'''Prototxt parser.

PrototxtParser is for parsing layer configurations from .prototxt file using
Google Protobuf API.

Note:
  The .prototxt layer must start with 'layer {...}'.
  The layer type must be string, as described in:
    http://caffe.berkeleyvision.org/tutorial/layers.html

Reference: https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py
'''
from __future__ import print_function

import os
import json
import numpy as np

from caffe.proto import caffe_pb2
from google.protobuf import text_format


class PrototxtParser:
    '''Parse layer configurations from a prototxt file.

    The layer and its parsed configurations:
    (1) Data/DummyData: [input_shape]
    (2) Convolution: [num_output, kW,kH,dW,dH,pW,pH]
    (3) Pooling: [pool_type, kW,kH,dW,dH,pW,pH],
          pool_type = (0=MAX, 1=AVE, 2=STOCHASTIC)
    (4) Dropout: [drop_ratio]
    (5) InnerProduct: [num_output]
    '''
    def __init__(self, prototxt):
        print('==> Parsing prototxt..')

        net = caffe_pb2.NetParameter()
        with open(prototxt, 'r') as f:
            text_format.Merge(f.read(), net)

        net_layers = net.layer
        num_layers = len(net_layers)
        self.layers = []

        # Input layer.
        input_layers = [l for l in net_layers if l.type.lower() in ['data', 'dummydata']]
        num_input_layers = len(input_layers)

        assert num_input_layers in [0,1], 'Net has %d input layers!' % num_input_layers

        if num_input_layers == 0:
            print('No data layer in prototxt, checking for input params..')
            assert len(net.input) > 0 and len(net.input_shape) > 0
            input_layer_name = net.input[0]
            input_shape = list(net.input_shape[0].dim)
            print('... Find input layer', input_layer_name, input_shape)

            # Add input layer.
            num_layers += 1
            self.layers.append({
                'name': input_layer_name,
                'type': 'DummyData',
                'input_shape': input_shape
            })

        # Map layer_name to index.
        name_to_index = {input_layer_name: 0} if num_input_layers == 0 else {}
        for layer in net_layers:
            layer_name = layer.name
            name_to_index[layer_name] = len(name_to_index)

        # Net structure graph represented in adjacent matrix.
        self.graph = np.zeros((num_layers, num_layers))

        # Add other net_layers.
        for i, layer in enumerate(net_layers):
            layer_name = layer.name
            layer_type = layer.type
            assert type(layer_type==str), 'ERROR: only string layer type supported!'
            print('... Find layer %s' % layer_name)

            # Add edge to graph.
            if layer_type.lower() not in ['data', 'dummydata']:
                cur_index = name_to_index[layer_name]
                btm_index = name_to_index[layer.bottom[0]]
                self.graph[btm_index][cur_index] = 1

            layer_config = {}
            if layer_type == 'Convolution':
                cfg = layer.convolution_param
                num_output = cfg.num_output
                kW = cfg.kernel_size[0] if len(cfg.kernel_size) else cfg.kernel_w
                kH = cfg.kernel_size[0] if len(cfg.kernel_size) else cfg.kernel_h
                dW = cfg.stride[0] if len(cfg.stride) else cfg.stride_w
                dH = cfg.stride[0] if len(cfg.stride) else cfg.stride_h
                pW = cfg.pad[0] if len(cfg.pad) else cfg.pad_w
                pH = cfg.pad[0] if len(cfg.pad) else cfg.pad_h
                dW = dW if dW else 1  # set default stride=1
                dH = dH if dH else 1
                layer_config = {'num_output': num_output,
                                'kW': kW,
                                'kH': kH,
                                'dW': dW,
                                'dH': dH,
                                'pW': pW,
                                'pH': pH}
            elif layer_type == 'Pooling':
                cfg = layer.pooling_param
                pool_type = cfg.pool  # MAX=0, AVE=1, STOCHASTIC=2
                kW = cfg.kernel_w if cfg.kernel_w else cfg.kernel_size
                kH = cfg.kernel_h if cfg.kernel_h else cfg.kernel_size
                dW = cfg.stride_w if cfg.stride_w else cfg.stride
                dH = cfg.stride_h if cfg.stride_h else cfg.stride
                pW = cfg.pad_w if cfg.pad_w else cfg.pad
                pH = cfg.pad_h if cfg.pad_h else cfg.pad
                dW = dW if dW else 1  # set default stride=1
                dH = dH if dH else 1
                layer_config = {'pool_type': pool_type,
                                'kW': kW,
                                'kH': kH,
                                'dW': dW,
                                'dH': dH,
                                'pW': pW,
                                'pH': pH}
            elif layer_type == 'Dropout':
                p = layer.dropout_param.dropout_ratio
                layer_config = {'dropout_ratio': p}
            elif layer_type == 'InnerProduct':
                num_output = layer.inner_product_param.num_output
                layer_config = {'num_output': num_output}

            layer_config.update({
                'name': layer_name,
                'type': layer_type,
            })

            self.layers.append(layer_config)

        self.save_config_and_graph()

    def save_config_and_graph(self):
        '''Save config and graph.'''
        CONFIG_DIR = './config/'
        if not os.path.isdir(CONFIG_DIR):
            os.mkdir(CONFIG_DIR)

        print('Saving net config..')
        with open(CONFIG_DIR + 'net.json', 'w') as f:
            json.dump(self.layers, f, indent=2)

        print('Saving graph..')
        np.save(CONFIG_DIR + 'graph.npy', self.graph)
