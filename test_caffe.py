import os
os.environ['GLOG_minloglevel'] = '2' # hide debug log

import caffe
import numpy as np

prototxt = './model/net.t7.prototxt'
binary = './model/net.t7.caffemodel'

caffe.set_mode_cpu()
net = caffe.Net(prototxt, binary, caffe.TEST)
net.blobs['data'].reshape(1,2,10,10)


x = np.load('x.npy')
# x = np.random.rand(1,1,5,5)
net.blobs['data'].data[...] = x
out = net.forward()
print(out)
