# -----------------------------------------------------------------------------
# Prototxt parser using Google Protobuf API
# Note only CONV and POOL layer params need to be parsed from prototxt.
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
            print('find layer '+layer_name)
            if layer_type == 'Convolution':
                kW = layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else layer.convolution_param.kernel_w
                kH = layer.convolution_param.kernel_size[1] if len(layer.convolution_param.kernel_size) else layer.convolution_param.kernel_h
                dW = layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else layer.convolution_param.stride_w
                dH = layer.convolution_param.stride[1] if len(layer.convolution_param.stride) else layer.convolution_param.stride_h
                pW = layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else layer.convolution_param.pad_w
                pH = layer.convolution_param.pad[1] if len(layer.convolution_param.pad) else layer.convolution_param.pad_h
                self.params[layer_name] = [kW,kH,dW,dH,pW,pH]
            elif layer_type == 'Pooling':
                # TODO
                pass

    def get_params(self, layer_name):
        '''Return layer params.'''
        return self.params[layer_name]

# p = PrototxtParser('./model/net.t7.prototxt')
# p.params
