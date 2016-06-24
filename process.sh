#!/usr/bin/env bash

# which model directory to go through
model_dir=model/latefusion1

# general model and directory params
gpuid=1
# proto_file=model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference_deploy.prototxt
proto_file=model/latefusion1/latefusion_inference_deploy.prototxt
image_dir="/data/akaspar/test/Images"
nextim_dir="/data/akaspar/test/NextImages"
group=$(basename $model_dir)
base_output_dir="/data/akaspar/test/$group"
images=$(ls $nextim_dir)

for model in $model_dir/*.caffemodel; do
  if [[ ! -f "$model" ]]; then
    echo "Invalid model $model!"
    exit 1
  fi
  echo "Testing model $model"
  model_name=$(basename "$model")
  model_name="${model_name/-inference.caffemodel/}"
  output_dir="$base_output_dir/$model_name"
  mkdir -p "$output_dir" 2>/dev/null
  # single image processing
  if [[ -z "$nextim_dir" ]]; then
    python << EOF
import decnet as dn

net = dn.init(proto = '$proto_file', weight = '$model', gpuid = $gpuid)
files = """$images""".split("\n")
for i,file in enumerate(files):
  print('[%d of %d] %s' %(i, len(files), file))
  dn.process(net, '$image_dir/' + file, '$output_dir/' + file)
EOF
  else
    # two-image processing
    python << EOF
import decnet as dn

net = dn.init(proto = '$proto_file', weight = '$model', gpuid = $gpuid)
files = """$images""".split("\n")
for i,file in enumerate(files):
  print('[%d of %d] %s' %(i, len(files), file))
  dn.process2(net, '$image_dir/' + file, '$nextim_dir/' + file, '$output_dir/' + file)
EOF
  fi
done
