#!/usr/bin/env bash

input_dir=/media/BigBertha/akaspar/Data/DecoupledNet/data/crops/inputs
input_next_dir=/media/BigBertha/akaspar/Data/DecoupledNet/data/crops/inputs_next
#model_dir=model/DecoupledNet_Full_anno
#model_dir=/media/BigBertha/akaspar/Data/DecoupledNet/models/Retrained/Soccer3
proto=/home/akaspar/Projects/decnet-scripts/model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference_deploy.prototxt
model_dir=/media/Mandelbrot/akaspar/Data/Models
base_output_dir=/media/Mandelbrot/akaspar/Data/Soccer/SuperLateFusion

shopt -s globstar

pushd ..

for model in $model_dir/**/*.caffemodel; do
  echo "Testing model $model"
  model_name=$(basename $(dirname "$model"))
  output_dir="$base_output_dir/$model_name"
  mkdir -p "$output_dir/next" 2>/dev/null
  read -r -d '' SCRIPT << EOM
decnet_init('$proto', '$model', 0);
files=dir('$input_next_dir');
for i = 1:numel(files),
    if files(i).isdir, continue; end;
    [~, name, ext] = fileparts(files(i).name);
    fprintf('[%d of %d] %s\n', i, numel(files), files(i).name);
    I = imread(fullfile('$input_next_dir', files(i).name));
    S = decnet_segment(I);
    imwrite(S, fullfile('$output_dir', 'next', files(i).name));
end;
exit; quit;
EOM
# @see http://stackoverflow.com/questions/19345872/dirty-variable-remove-carriage-return
  SCRIPT=$(echo "$SCRIPT" | tr '\n' ' ')
# or SCRIPT="${SCRIPT//[$'\n']/ }"
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/lib matlab -nojvm -nodesktop -r "$SCRIPT"
  echo "Result in $output_dir"
done

popd
