#---------------------------------------------------------------
# Decompose layer params of a caffemodel to disk.
#---------------------------------------------------------------
import os
import caffe
import numpy as np
from prototxt_parser import PrototxtParser


def writeln(file, lst):
    '''Write list content to file'''
    content = ''
    for x in lst:
        content = content + str(x) + '\t'
    file.write(content+'\n')

def println(lst):
    '''Print list content to console'''
    content = ''
    for x in lst:
        content = content + str(x) + ' '
    print(content)

# 1. define .prototxt parser
parser = PrototxtParser('./model/net.t7.prototxt')

# 2. load caffe model
net = caffe.Net('./model/net.t7.prototxt', './model/net.t7.caffemodel', caffe.TEST)

# directory for saving layer params and log
save_dir = './params/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

logfile = open(save_dir+'net.log', 'w')

# 3. parse layer params
print('\nexporting..')
layer_num = len(net.layers)
for i in range(layer_num):
    layer_type = net.layers[i].type
    layer_name = net._layer_names[i]

    if layer_type == 'InnerProduct':
        layer_weight = net.params[layer_name][0].data
        layer_bias = net.params[layer_name][1].data
        input_size = layer_weight.shape[1]
        output_size = layer_weight.shape[0]
        # save params
        np.save(save_dir+layer_name+'_weight', layer_weight)
        np.save(save_dir+layer_name+'_bias', layer_bias)
        # logging
        println(['==> layer', i, ': Linear [', str(input_size), '->', str(output_size), ']'])
        writeln(logfile, [i, 'Linear', layer_name])
    elif layer_type == 'ReLU':
        print('==> layer '+str(i)+' : ReLU')
        writeln(logfile, [i, 'ReLU', layer_name])
    elif layer_type == 'Flatten':
        print('==> layer '+str(i)+' : Flatten')
        writeln(logfile, [i, 'Flatten', layer_name])
    elif layer_type == 'Convolution':
        layer_weight = net.params[layer_name][0].data
        layer_bias = net.params[layer_name][1].data
        np.save(save_dir+layer_name+'_weight', layer_weight)
        np.save(save_dir+layer_name+'_bias', layer_bias)
        # get CONV params
        kW,kH,dW,dH,pW,pH = parser.get_params(layer_name)
        # logging
        println(['==> layer', i, ': Convolution [', kW,kH,dW,dH,pW,pH, ']'])
        writeln(logfile, [i, 'Convolution', layer_name, kW,kH,dW,dH,pW,pH])
    elif layer_type == 'BatchNorm':
        running_mean = net.params[layer_name][0].data
        running_var = net.params[layer_name][1].data
        # momentum = net.params[layer_name][2].data
        np.save(save_dir+layer_name+'_mean', running_mean)
        np.save(save_dir+layer_name+'_var', running_var)
        print('==> layer '+str(i)+' : BatchNorm')
        writeln(logfile, [i, 'BatchNorm', layer_name])

logfile.close()







#
# net._layer_names[2]
# net.layers[2].type
#
# name = net._layer_names[2]
#
# running_mean = net.params[name][0].data
# running_var = net.params[name][1].data
# momentum = net.params[name][2].data



#
# w.shape
# b.shape
#
# name
#
# net.blobs[name]
