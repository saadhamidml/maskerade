#!/bin/bash

set +e

./repeat_runs.sh -c configs/airpass/bq_r.yaml -s seeds.txt -x 10
./repeat_runs.sh -c configs/airpass/bq_u.yaml -s seeds.txt -x 10
./repeat_runs.sh -c configs/airpass/bq.yaml -s seeds.txt -x 10

./repeat_runs.sh -c configs/mauna_loa/bq_r.yaml -s seeds.txt -x 10
./repeat_runs.sh -c configs/mauna_loa/bq_u.yaml -s seeds.txt -x 10
./repeat_runs.sh -c configs/mauna_loa/bq.yaml -s seeds.txt -x 10
