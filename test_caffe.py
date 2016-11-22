import os
os.environ['GLOG_minloglevel'] = '2' # hide debug log

import caffe
import numpy as np

prototxt = './cvt_net.prototxt'
binary = './cvt_net.caffemodel'

# prototxt = '/mnt/hgfs/D/download/vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt'
# binary = '/mnt/hgfs/D/download/vgg_face_caffe/vgg_face_caffe/VGG_FACE.caffemodel'
# prototxt = '/home/luke/workspace/child/model/child.prototxt'
# binary = '/home/luke/workspace/child/model/child.caffemodel'

caffe.set_mode_cpu()
net = caffe.Net(prototxt, binary, caffe.TEST)
net.blobs['data'].reshape(1,3,96,96)

x = np.load('x.npy')
# x = np.random.rand(1,3,96,96)
# np.save('x.npy',x)
net.blobs['data'].data[...] = x
out = net.forward()
print(out)
# np.save('y_caffe.npy', out['linear55'])
# f = net.blobs['conv1'].data
# print(f)
