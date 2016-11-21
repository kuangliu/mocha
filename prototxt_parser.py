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

from caffe.proto import caffe_pb2
from google.protobuf import text_format
from google.protobuf.descriptor import FieldDescriptor as FD


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

        layers = net.layer
        if len(layers) == 0:
            raise NotImplementedError('Caffemodel prototxt has 0 layers!')

        self.layer_config = {}

        # Data layer.
        data_layers = [l for l in layers if l.type.lower() in ['data', 'dummydata']]

        num_data_layers = len(data_layers)
        if num_data_layers == 0:
            print('No data layer in prototxt, checking for input params..')

            assert len(net.input) > 0
            assert len(net.input_shape) > 0

            input_layer_name = net.input[0]
            input_shape = net.input_shape[0].dim
            print('... Find input layer', net.input[0], input_shape)

            self.input_layer_name = input_layer_name
            self.layer_config[input_layer_name] = {'input_shape': list(input_shape)}
        elif num_data_layers == 1:
            data_layer = data_layers[0]
            layer_name = data_layer.name

            self.input_layer_name = layer_name
            self.layer_config[layer_name] = {'input_shape': list(data_layer.dummy_data_param.shape[0].dim)}
        else:
            raise NotImplementedError('Number of data layers %d != 1' % num_data_layers)

        # Other layers.
        for layer in layers:
            layer_name = layer.name
            layer_type = layer.type

            assert type(layer_type==str), 'ERROR: only string layer type supported!'
            print('... Find layer ', layer_name)

            if layer_type == 'Convolution':
                cfg = layer.convolution_param
                num_output = cfg.num_output
                kW = cfg.kernel_size[0] if len(cfg.kernel_size) else cfg.kernel_w
                kH = cfg.kernel_size[0] if len(cfg.kernel_size) else cfg.kernel_h
                dW = cfg.stride[0] if len(cfg.stride) else cfg.stride_w
                dH = cfg.stride[0] if len(cfg.stride) else cfg.stride_h
                pW = cfg.pad[0] if len(cfg.pad) else cfg.pad_w
                pH = cfg.pad[0] if len(cfg.pad) else cfg.pad_h
                dW = dW if dW else 1    # set default stride=1
                dH = dH if dH else 1
                self.layer_config[layer_name] = {'num_output': num_output,
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
                dW = dW if dW else 1    # set default stride=1
                dH = dH if dH else 1
                self.layer_config[layer_name] = {'pool_type': pool_type,
                                           'kW': kW,
                                           'kH': kH,
                                           'dW': dW,
                                           'dH': dH,
                                           'pW': pW,
                                           'pH': pH}
            elif layer_type == 'Dropout':
                p = layer.dropout_param.dropout_ratio
                self.layer_config[layer_name] = {'dropout_ratio': p}
            elif layer_type == 'InnerProduct':
                num_output = layer.inner_product_param.num_output
                self.layer_config[layer_name] = {'num_output': num_output}

    def get_layer_config(self, layer_name):
        '''Return layer config.'''
        c = self.layer_config.get(layer_name)
        return c if c else {}
