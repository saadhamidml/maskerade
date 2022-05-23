#!/bin/bash

set +e

./repeat_runs.sh -c configs/mauna_loa/bq.yaml -s seeds.txt -x 10
./repeat_runs.sh -c configs/mauna_loa/bq_u.yaml -s seeds.txt -x 10