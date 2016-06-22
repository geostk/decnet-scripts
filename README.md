# decnet-scripts
Scripts and personal models for DecoupledNet, specifically targetting the "person" class.

## Matlab
Most scripts are for matlab and based on the code from DecoupledNet.

* `decnet_init(proto, model, use_gpu)` - initialize the decnet (global)
* `decnet_segment(img, marginal, thresh, ref_size)` - compute the segmentation of an image using decnet, marginal corresponds to the type of marginalization
* `decnet_pyramid(img, levels, overlap, ref_size)` - compute a segmentation using a pyramid decomposition of the image into tiles of size ref_size on multiple scales

## Python
See `demo.py` for the basic usage:
```
#!/usr/bin/env python

import decnet as dn

net = dn.init(use_gpu = False)
dn.process(net, 'example.png', 'segmentation.png')
```

The functions are in `decnet.py`:

* `init(proto, weight, use_gpu, gpuid)` - load and return a Caffe net object
* `process(net, input_file, output_file, ref_size = 320)` - segment an image and save the output
* `segment(net, img, ref_size = 320)` - compute the segmentation of a BGR image (cv2)

## Directories
The scripts in 

* `batch` - some kind of batch processing (data generation, evaluation, testing)
* `eval` - for evaluating segmentation results
* `other` - tentative segmentation scripts, not used
* `util` - helper functions that are useful

