#!/bin/bash

IN_DIR="/mnt/beegfs/home/mcastel1/$1/solution"
OUT_DIR=$2

echo "Input directory = " $IN_DIR
echo "Output directory = " $OUT_DIR

rsync -chavzP --stats mcastel1@abacus:$IN_DIR $OUT_DIR
