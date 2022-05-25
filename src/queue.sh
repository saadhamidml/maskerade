#!/bin/bash

set +e


./repeat_runs.sh -c configs/airpass/bq.yaml -r 1 -x 10

./repeat_runs.sh -c configs/mauna_loa/bq.yaml -s seeds.txt -x 10
