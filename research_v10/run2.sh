#!/bin/bash
cd "$(dirname "$0")"
{ for r in 0 1 2 3 4; do for v in C O B set3; do echo "$v ${v}_r$r - $r"; done; done
  for r in 0 1 2 3 4; do echo "B cap84_B_r$r - $r cap"; done; } |
xargs -P 2 -L 1 bash -c 'if [ "$4" = cap ]; then export MAXDAY=84; fi; OMP_NUM_THREADS=1 python3 seq_eval.py $0 $1 $2 $3 > log_$1.txt 2>&1; tail -1 log_$1.txt'
echo ALLDONE
