#!/usr/bin/env bash

image_dir=/media/Mandelbrot/akaspar/Data/Soccer/Test/Images/
#model_dir=model/DecoupledNet_Full_anno
#model_dir=/media/BigBertha/akaspar/Data/DecoupledNet/models/Retrained/Soccer3
model_dir=/media/Mandelbrot/akaspar/Data/Models/DecoupledNet_Full_anno
group=$(basename $model_dir)
base_output_dir=/media/Mandelbrot/akaspar/Data/Soccer/Test/$group

for model in $model_dir/*.caffemodel; do
  echo "Testing model $model"
  model_name=$(basename "$model")
  model_name="${model_name/-inference.caffemodel/}"
  output_dir="$base_output_dir/$model_name"
  mkdir -p "$output_dir/.cmaps" 2>/dev/null
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/lib matlab -nojvm -nodesktop -r "decnet_init([], '$model', 0); files=dir('$image_dir'); for i = 1:numel(files); if files(i).isdir, continue; end; fprintf('[%d of %d] %s\n', i, numel(files), files(i).name); I = imread(fullfile('$image_dir', files(i).name)); S = decnet_segment(I); L = labelize(S, 256); imwrite(S, fullfile('$output_dir', files(i).name)); imwrite(L, parula(256), fullfile('$output_dir', '.cmaps', ['_' files(i).name])); end; exit; quit;"
  echo "Result in $output_dir + .cmaps/.human/.background"
done
