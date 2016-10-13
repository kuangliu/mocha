#---------------------------------------------------------------
# Decompose layer params of a caffemodel to disk.
#---------------------------------------------------------------
import os
import caffe
import numpy as np
from parser import PrototxtParser


#---------------------------------------------------------------
# 1. define .prototxt parser
#
parser = PrototxtParser('./model/net.t7.prototxt')

# 2. load caffe model
net = caffe.Net('./model/net.t7.prototxt', './model/net.t7.caffemodel', caffe.TEST)

# directory for saving layer params (weight + bias)
save_dir = './params/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# log file storing layer info
net_log = open(save_dir+'net.log', 'w')

# 3. dump layer params
print('\nexporting..')
layer_num = len(net.layers)
for i in range(layer_num):
    layer_type = net.layers[i].type
    layer_name = net._layer_names[i]

    if layer_type == 'InnerProduct':
        layer_weight = net.params[layer_name][0].data
        layer_bias = net.params[layer_name][1].data
        # save params
        np.save(save_dir+layer_name+'_weight', layer_weight)
        np.save(save_dir+layer_name+'_bias', layer_bias)
        # logging
        net_log.write(str(i)+'\tLinear\t'+layer_name+'\n')
        # print info
        input_size = layer_weight.shape[1]
        output_size = layer_weight.shape[0]
        print('==> layer '+str(i)+': Linear ['+str(input_size)+'->'+str(output_size)+']')

    elif layer_type == 'ReLU':
        print('==> layer '+str(i)+': ReLU')
        net_log.write(str(i)+'\tReLU\t'+layer_name+'\n')

    elif layer_type == 'Convolution':
        print('==> layer '+str(i)+': Convolution')




net_log.close()


# net._layer_names[0]
# name = net._layer_names[1]
#
#
# w = net.params[name][0].data
# b = net.params[name][1].data
#
# w.shape
# b.shape
#
# name
#
# net.blobs[name]
