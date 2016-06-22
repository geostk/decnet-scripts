#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('./caffe/python')
import caffe
from PIL import Image
from caffe.proto import caffe_pb2
import cv2

def init( proto = None, weight = None, use_gpu = True, gpuid = 0 ):
    if proto is None:
        proto = 'model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference_deploy.prototxt'
    if weight is None:
        weight = 'model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference.caffemodel'
    net = caffe.Net(proto, weight)
    if use_gpu:
        net.set_mode_gpu()
        net.set_device(gpuid)
    else:
        net.set_mode_cpu()
    net.set_phase_test()
    return net

def preprocess( img, ref_sz ):
    # images for network are [1, 3, h, w]
    # but we use [w, h, 3] here
    meanImg = np.array([[[104.00698793, 116.66876762, 122.67891434]]]) # order = bgr
    meanImg = np.tile(meanImg, [ref_sz, ref_sz, 1])
    resized = cv2.resize(img, (ref_sz, ref_sz))
    return resized - meanImg

def process( net, inp_file, out_file, ref_size = 320 ):
    im = cv2.imread(inp_file)
    out,_ = segment(net, im, ref_size)
    cv2.imwrite(out_file, (out * 255).astype(np.uint8))
    return out

def segment( net, img, ref_size = 320 ):
    img = np.array(img, dtype=float)
    h, w, ch = img.shape
    
    # pad image to be square
    im_sz = max(h, w)
    caffe_im = np.lib.pad(img, [(0, im_sz - h), (0, im_sz - w), (0, 0)], 'constant', constant_values = 0)
    # caffe_im = padarray(img, [im_sz - h, im_sz - w], 'post')
    caffe_im = preprocess(caffe_im, ref_size)
    # conversion to [1, 3, h, w] format
    # XXX is it [1,3,h,w] or [1,3,w,h]
    caffe_im = np.transpose(caffe_im, [2, 0, 1])
    caffe_im = caffe_im[np.newaxis, :, :, :]

    # pre-computation depending on the marginalization method
    label = np.zeros([1,20,1,1])
    label[0,14,0,0] = 1
    
    # compute background and classification
    net.blobs['data'].data[...] = caffe_im
    net.blobs['cls-score-masked'].data[...] = label
    net.forward()
    
    cls_score = np.squeeze(net.blobs['cls-score-pooled'].data)
    # cls_score_max = cnn_output{2};
    seg_score = np.squeeze(net.blobs['seg-score'].data)
    
    # marginalization
    softmax_score = np.exp(seg_score - np.tile(np.amax(seg_score, axis=0), [2, 1, 1]))
    softmax_score = np.divide(softmax_score, np.tile(np.sum(softmax_score, axis=0), [2, 1, 1]))

    # background = np.squeeze(softmax_score[0,:,:]);
    human = np.squeeze(softmax_score[1,:,:])

    resize_score_map = cv2.resize(human, (im_sz, im_sz))
    return resize_score_map[0:h, 0:w],cls_score

