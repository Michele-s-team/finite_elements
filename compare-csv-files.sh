#!/bin/bash


# run with ./compare-csv-files.sh steady-state-no-flow/solution-old steady-state-no-flow/solution-new
PATH_A=$1
PATH_B=$2

find "$PATH_A" -name '*.csv' -exec bash -c 'diff "$1" "$2/${1#"$3"/}"' _ {} "$PATH_B" "$PATH_A" \;
