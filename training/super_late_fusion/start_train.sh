LOGDIR=./training_log
CAFFE=./caffe/.build/tools/caffe
SOLVER=./solver.prototxt
#WEIGHTS=./fp_bp_model/fp_bp_model.caffemodel
#WEIGHTS=../../model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference.caffemodel

PARAMS="-solver $SOLVER -gpu 0"
if [[ -n "$RESUME" ]]; then
  PARAMS="$PARAMS --snapshot=$RESUME"
else
  echo "From scratch"
  # PARAMS="$PARAMS -weights $WEIGHTS" # no initial weight for super-late fusion
fi

if [[ "$DEBUG" -ne 1 ]]; then
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 GLOG_log_dir=$LOGDIR ${CAFFE/.build/.build_release} train $PARAMS
elif [[ "$VERBOSE" -eq 1 ]]; then
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 GLOG_log_dir=$LOGDIR ${CAFFE/.build/.build_debug} train -v $PARAMS
else
  echo "run train $PARAMS"
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 GLOG_log_dir=$LOGDIR gdb ${CAFFE/.build/.build_debug}
fi
