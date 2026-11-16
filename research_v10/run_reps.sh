#!/bin/bash
# 5 outer repeats x 4 variants, two at a time (2 CPUs)
cd "$(dirname "$0")"
jobs=()
for r in 0 1 2 3 4; do for v in C O B set3; do jobs+=("$v $r"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -L 1 bash -c 'OMP_NUM_THREADS=1 python3 seq_eval.py $0 ${0}_r$1 - $1 > log_${0}_r$1.txt 2>&1; tail -1 log_${0}_r$1.txt'
echo ALLDONE
