# -----------------------------------------------------------------------------
# PrototxtParser is for parsing CONV/POOL params from .prototxt file
# using Google Protobuf API.
#
# ref: https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py
# -----------------------------------------------------------------------------

from caffe.proto import caffe_pb2
from google.protobuf import text_format


class PrototxtParser:
    '''Load a prototxt file and parse CONV/POOL params out.'''
    def __init__(self, prototxt):
        print('==> parse prototxt..')
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt,'r').read(), net)
        self.params = {}

        for layer in net.layer:
            layer_name = layer.name
            layer_type = layer.type
            print('find layer '+layer_name+' '+layer_type)
            if layer_type == 'Convolution':
                param = layer.convolution_param
                print(param.kernel_size)
                kW = param.kernel_size[0] if len(param.kernel_size) else param.kernel_w
                kH = param.kernel_size[0] if len(param.kernel_size) else param.kernel_h
                dW = param.stride[0] if len(param.stride) else param.stride_w
                dH = param.stride[0] if len(param.stride) else param.stride_h
                pW = param.pad[0] if len(param.pad) else param.pad_w
                pH = param.pad[0] if len(param.pad) else param.pad_h
                self.params[layer_name] = [kW,kH,dW,dH,pW,pH]
            elif layer_type == 'Pooling':
                param = layer.pooling_param
                pool_type = param.pool  # MAX=0, AVE=1, STOCHASTIC=2
                kW = param.kernel_size if param.kernel_size else param.kernel_w
                kH = param.kernel_size if param.kernel_size else param.kernel_h
                dW = param.stride if param.stride else param.stride_w
                dH = param.stride if param.stride else param.stride_h
                pW = param.pad if param.pad else param.pad_w
                pH = param.pad if param.pad else param.pad_h
                self.params[layer_name] = [pool_type,kW,kH,dW,dH,pW,pH]

    def get_params(self, layer_name):
        '''Return layer params.'''
        return self.params[layer_name]


# prototxt = './model/net.t7.prototxt'
# p = PrototxtParser('./model/net.t7.prototxt')
# p.params

# net.layer[0].name
# net.layer[1].name
# net.layer[2].name
# net.layer[3].name
# net.layer[4].name
#
# layer = net.layer[3]
# layer.

# layer.pooling_param.AVE
