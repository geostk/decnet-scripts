#!/bin/bash
ROOT=../caffe
THIS=$(dirname $0)

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename $0) log_file"
  exit 1
fi

# name of logfile
LOGNAME=$1
LOGTHIS="$THIS/$(basename $LOGNAME)"

if [[ ! -f "$LOGNAME" ]]; then
  echo "No log file found: $LOGNAME"
  exit 2
fi

# extract information
${ROOT}/tools/extra/parse_log.sh "${LOGNAME}"

# plot everything
gnuplot <<- EOF
  reset
  set key right bottom
  set style data lines
  set font 'Consolas,10'
  set term png enhanced font 'Consolas,10' size 800,400
  set title "Learning process ${LOGNAME}"
  set xlabel "Iterations"
  set ylabel "Loss"  tc rgb "#009e73"
  set y2label "Accuracy" tc rgb "#7570b3"
  set output "run.png"
  set border 11 back ls 80 lc -1
  set mxtics 2
  set grid mxtics
  set ytics autofreq tc rgb "#009e73"
  set y2tics autofreq tc rgb "#7570b3"
  set logscale yy2
  set grid xtics
  set grid ytics

  plot "${LOGTHIS}.train" using 1:3 title "training (loss)" lt 2 lc rgb "#aaaaaa" axes x1y1,  \
  "${LOGTHIS}.test" using 1:4 title "test (loss)" lt 2 lc rgb "#009e73" lw 2 axes x1y1  ,\
  "${LOGTHIS}.test" using 1:3 title "test (accuracy)" lt 2 lc rgb "#7570b3" lw 2 axes x1y2 
EOF
cp run.png run-$(date +"%T").png
