#!/usr/bin/env bash

#caffe_model = script_path + '/002_train_seg_Full_anno_make_inference.prototxt'
#caffe_weight = script_path + '/snapshot/002_train_seg_Full_anno_iter_last.caffemodel'

INFER_INPUT=superlatefusion_iter_last.caffemodel
INFER_OUTPUT=superlatefusion_inference.caffemodel
OUTPUT_DIR=/media/Amaterasu/akaspar/Data/Soccer/Models/SuperLateFusionResidual_from_decnet
# OUTPUT_DIR=/media/Izanagi/akaspar/Data/Soccer/Models/Original3

MODELS=$(ls snapshot/superlatefusion_iter_[1-9]*.caffemodel)
MODELS="snapshot/superlatefusion_iter_15000.caffemodel snapshot/superlatefusion_iter_10000.caffemodel snapshot/superlatefusion_iter_500.caffemodel snapshot/superlatefusion_iter_1000.caffemodel snapshot/superlatefusion_iter_14500.caffemodel snapshot/superlatefusion_iter_10500.caffemodel snapshot/superlatefusion_iter_9500.caffemodel snapshot/superlatefusion_iter_1500.caffemodel"

echo "Models: $MODELS"

for iter in $MODELS; do
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
  cp $INFER_OUTPUT $TARGET
  echo "Copied inference to $TARGET"
done
