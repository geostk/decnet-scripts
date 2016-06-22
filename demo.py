#!/usr/bin/env python

import decnet as dn

net = dn.init(use_gpu = False)
dn.process(net, 'example.png', 'segmentation.png')
