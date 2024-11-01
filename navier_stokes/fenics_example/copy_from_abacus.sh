#ths script copies from abacus all files of typs $2 from folder $1 into folder $3, by removing $3 and creating a new one
#run with
# ./copy_from_abacus.sh [folder where to read the files on abacus] [type of files to read] [folder where to store the files]
# ./copy_from_abacus.sh dynamics-bcs-w-square-1/solution/snapshots/h5 z_\* solution/snapshots/h5

#!/bin/bash

IN_DIR="/mnt/beegfs/home/mcastel1/$1"
FILES_TO_COPY=$2
OUT_DIR=$3


clear; clear;

echo "Input directory = " $IN_DIR
echo "Files to copy = " $FILES_TO_COPY
echo "Output directory = " $OUT_DIR

rm -rf $OUT_DIR
mkdir $OUT_DIR

rsync --stats --size-only -P -v -e ssh mcastel1@abacus:$IN_DIR/$FILES_TO_COPY $OUT_DIR

