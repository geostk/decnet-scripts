#!/usr/bin/env bash

input=example.png
model_dir=/media/BigBertha/akaspar/Data/DecoupledNet/models/Retrained/Soccer5

for model in $model_dir/*.caffemodel; do
  if [[ -f "${model}_segment.png" ]]; then
    echo "Skipping $model"
    continue
  fi
  echo "Testing model $model"
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/lib matlab -nojvm -nodesktop -r "decnet_init([], '$model', 0); I = imread('$input'); S = decnet_segment(I); L = labelize(S, 256); imwrite(S, '${model}_segment.png'); imwrite(L, parula(256), '${model}_segment-cmap.png'); exit; quit;"
  echo "Result in ${model}_segment.png"
done
