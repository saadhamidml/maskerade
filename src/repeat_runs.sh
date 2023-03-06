#!/bin/bash

cd "${0%/*}"

# Assign flags to variables
while getopts c:r:s:x:t: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        r) repeats=${OPTARG};;
        s) seeds=${OPTARG};;
        x) crossvalidate=${OPTARG};;
    esac
done

problem_sans_method="${config%/*}"
problem="${problem_sans_method##*/}"
method_sans_yaml="${config##*/}"
method="${method_sans_yaml%.*}"
datetime=$(date +"%m%d%H%M")
collection_name="collection_${problem}_${method}_${datetime}"

set +e

if [ -n "$repeats" ]; then
  for i in $(seq 1 $repeats)
  do
    if [ -n "$crossvalidate" ]; then
      seed=$RANDOM
      for i in $(seq 1 $crossvalidate)
      do
        python -u main.py with $config cross_validate=$i seed=$seed collection=$collection_name --force
      done
    else
      python -u main.py with $config collection=$collection_name --force
    fi
  done
fi

if [ -n "$seeds" ]; then
  if [ -n "$crossvalidate" ]; then
    for i in $(seq 1 $crossvalidate)
    do
      python -u main.py with $config cross_validate=$i seed=$seeds collection=$collection_name --force
    done
  else
    python -u main.py with $config seed=$seeds collection=$collection_name --force
  fi
fi

python metric_statistics.py $collection_name
