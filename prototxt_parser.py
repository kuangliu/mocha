# -----------------------------------------------------------------------------
# PrototxtParser is for parsing layer configs from .prototxt file using
# Google Protobuf API.
#
# The .prototxt layer must start with 'layer {...}'.
# The layer type must be string, as described in:
#   http://caffe.berkeleyvision.org/tutorial/layers.html
#
# ref: https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py
# -----------------------------------------------------------------------------

from caffe.proto import caffe_pb2
from google.protobuf import text_format


class PrototxtParser:
    '''Load a prototxt file and parse CONV/POOL/DROPOUT configs out.

    Returns:
     - CONV: [num_output, kW,kH,dW,dH,pW,pH]
     - POOLING: [pool_type, kW,kH,dW,dH,pW,pH]
        - pool_type = (0=MAX, 1=AVE, 2=STOCHASTIC)
     - DROPOUT: [drop_ratio]
     - InnerProduct: [num_output]
    '''
    def __init__(self, prototxt):
        print('==> parse prototxt..')
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt,'r').read(), net)

        self.config = {}
        for layer in net.layer:
            layer_name = layer.name
            layer_type = layer.type
            assert type(layer_type==str), 'Only string layer type supported!'
            print('... find layer '+layer_name)

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
                self.config[layer_name] = [num_output,kW,kH,dW,dH,pW,pH]
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
                self.config[layer_name] = [pool_type,kW,kH,dW,dH,pW,pH]
            elif layer_type == 'Dropout':
                p = layer.dropout_param.dropout_ratio
                self.config[layer_name] = [p]
            elif layer_type == 'InnerProduct':
                num_output = layer.inner_product_param.num_output
                self.config[layer_name] = [num_output]

    def get_config(self, layer_name):
        '''Return layer config.'''
        return self.config[layer_name]
