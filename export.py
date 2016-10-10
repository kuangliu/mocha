#################################################################
## Decompose layer params of a caffemodel to disk.
#################################################################
import os
import caffe
import numpy as np

# 1. load caffe model
net = caffe.Net('./model/net.t7.prototxt', './model/net.t7.caffemodel', caffe.TEST)

# directory for saving layer params (weight + bias)
save_dir = './params/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# log file storing layer info
net_log = open(save_dir+'net.log', 'w')

# 2. loop all layers
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
        # write log
        net_log.write(str(i)+'\t'+layer_type+'\t'+layer_name+'\n')

net_log.close()
