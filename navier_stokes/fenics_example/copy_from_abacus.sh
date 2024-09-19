#!/bin/bash

IN_DIR="/mnt/beegfs/home/mcastel1/$1/solution"
OUT_DIR=$2

echo "Input directory = " $IN_DIR
echo "Output directory = " $OUT_DIR

rsync --size-only -P -v -e ssh mcastel1@abacus:$IN_DIR/z_n.\* $OUT_DIR
