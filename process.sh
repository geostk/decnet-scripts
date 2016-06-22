#!/usr/bin/env bash

# which model directory to go through
model_dir=model/crop+pyramid1

# general model and directory params
gpuid=0
proto_file=model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference_deploy.prototxt
image_dir="/data/akaspar/test/Images"
group=$(basename $model_dir)
base_output_dir="/data/akaspar/test/$group"
images=$(ls $image_dir)

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
  python << EOF
import decnet as dn

net = dn.init(proto = '$proto_file', weight = '$model', gpuid = $gpuid)
files = """$images""".split("\n")
for i,file in enumerate(files):
  print('[%d of %d] %s' %(i, len(files), file))
  dn.process(net, '$image_dir/' + file, '$output_dir/' + file)
EOF
done
