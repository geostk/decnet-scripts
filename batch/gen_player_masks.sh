#!/usr/bin/env bash

if [[ $# -eq 1 ]]; then
  BASE_DIR="$1"
elif [[ $# -gt 1 ]]; then
  echo "Usage: ./$(basename $0) [basedir]"
  exit 1
else
  BASE_DIR=/media/Mandelbrot/akaspar/Tmp/Sequences
fi

PROTO=None
MODEL=/media/Mandelbrot/akaspar/Data/Models/Soccer5/soccer_iter_28000-inference.caffemodel
GPUID=0

PLAYER_DIR=player-soccer5

images=$(find /media/Mandelbrot/akaspar/Tmp/Sequences -path */image*.png -not -path */player*)
# create directories
for img in $images; do
  dir=$(dirname "$img")/"$PLAYER_DIR"
  if [[ ! -d "$dir" ]]; then
    mkdir "$dir"
  fi
done

python << EOF
import decnet as dn
import os

net = dn.init(proto = None, weight = '$MODEL', gpuid = $GPUID)
files = """$images""".split("\n")
for i,file in enumerate(files):
  dir=os.path.dirname(file)
  name=os.path.basename(file)
  target_file = os.path.join(dir, '$PLAYER_DIR', name)
  if os.path.exists(target_file):
    print('[%d of %d] %s already processed' %(i, len(files), file))
  else:
    print('[%d of %d] %s' %(i, len(files), file))
    dn.process(net, file, target_file)
EOF


