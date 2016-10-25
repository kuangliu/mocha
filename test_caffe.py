import os
os.environ['GLOG_minloglevel'] = '2' # hide debug log

import caffe
import numpy as np

prototxt = './model/net.prototxt'
binary = './model/net.caffemodel'
# prototxt = '/mnt/hgfs/D/download/vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt'
# binary = '/mnt/hgfs/D/download/vgg_face_caffe/vgg_face_caffe/VGG_FACE.caffemodel'
# prototxt = '/home/luke/workspace/child/model/child.prototxt'
# binary = '/home/luke/workspace/child/model/child.caffemodel'

caffe.set_mode_cpu()
net = caffe.Net(prototxt, binary, caffe.TEST)
net.blobs['data'].reshape(1,1,28,28)


x = np.load('x.npy')
# x = np.random.rand(1,1,28,28)
# np.save('x.npy',x)
net.blobs['data'].data[...] = x
out = net.forward()
print(out)
np.save('y_caffe.npy', out['loss'])
