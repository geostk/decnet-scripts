#!/usr/bin/env bash

#caffe_model = script_path + '/002_train_seg_Full_anno_make_inference.prototxt'
#caffe_weight = script_path + '/snapshot/002_train_seg_Full_anno_iter_last.caffemodel'

INFER_INPUT=latefusion_iter_last.caffemodel
INFER_OUTPUT=latefusion_inference.caffemodel
OUTPUT_DIR=/data/akaspar/model/latefusion1
# OUTPUT_DIR=/media/Izanagi/akaspar/Data/Soccer/Models/Original3

for iter in snapshot/latefusion_iter_[1-9]*.caffemodel; do
  fname=$(basename $iter)
  TARGET=$OUTPUT_DIR/${fname/.caffemodel/-inference.caffemodel}
  if [[ -f $TARGET ]]; then
    echo "Skipping $iter"
    continue
  fi
  echo "Processing $iter"
  rm -rf snapshot/$INFER_INPUT
  rm -rf $INFER_OUTPUT
  pushd snapshot
  ln -s $(basename $iter) $INFER_INPUT
  popd
  ./BN_make_INFERENCE_script.py
  cp $INFER_OUTPUT $TARGET && echo "Copied inference to $TARGET"
done
