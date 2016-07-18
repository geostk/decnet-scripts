#!/usr/bin/env python

## path configuration
caffe_root = './caffe'
script_path = '.'
caffe_model = script_path + '/soccer_make_inference.prototxt'
caffe_weight = script_path + '/snapshot/superlatefusion_iter_last.caffemodel'
caffe_inference_weight = script_path + '/superlatefusion_inference.caffemodel'

### start generate caffemodel

print 'start generating BN-testable caffemodel'
print 'caffe_root: %s' % caffe_root
print 'script_path: %s' % script_path
print 'caffe_model: %s' % caffe_model
print 'caffe_weight: %s' % caffe_weight
print 'caffe_inference_weight: %s' % caffe_inference_weight


import numpy as np


import sys
sys.path.append(caffe_root+'/python')
import caffe
from caffe.proto import caffe_pb2

import cv2

net = caffe.Net(caffe_model, caffe_weight)
net.set_mode_cpu()
net.set_phase_test()


def forward_once(net):
    start_ind = 0
    end_ind = len(net.layers) - 1
    net._forward(start_ind, end_ind)
    return {out: net.blobs[out].data for out in net.outputs}


print net.params.keys()


res = forward_once(net)

layers = ['data', 'data2', 'seg-label', 'input', 'conv1', 'conv2', 'conv3', 'seg-score']

# debug output
for name in layers:
    blob = net.blobs[name]
    for b in range(0, 16):
        for c in range(0, blob.data[b].shape[0]):
            cv2.imwrite('debug/%s/%d-%d.png' %(name, b, c), np.squeeze(blob.data[b][c]))


print 'done'










