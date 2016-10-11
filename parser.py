#---------------------------------------------------------------
# Parse .prototxt file to get layer configuration parameters.
#---------------------------------------------------------------
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

class PrototxtParser:
    def __init__(self, prototxt):
        self.prototxt = prototxt
        f = open(prototxt, 'r')
        msg = caffe_pb2.NetParameter()
        text_format.Merge(f.read(), msg)
        # text_format.MessageToString(msg)
        # parse layer config to dict
        params = {}
        for _,field in msg.ListFields():
            for v in field:
                if type(v) == caffe_pb2.BlobShape:  # input_shape
                    pass
                    # print(k.dim)
                elif type(v) == caffe_pb2.LayerParameter:  # layer
                    print(v.convolution_param)

    def get_param(layer_name, param_name):
        return self.params[layer_name][param_name]
