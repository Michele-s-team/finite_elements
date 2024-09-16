#!/bin/bash

IN_DIR="/mnt/beegfs/home/mcastel1/fenics_example/solution"
OUT_DIR="/Users/michelecastellana/Desktop"

rsync --size-only -P -v -e ssh mcastel1@abacus:$IN_DIR/z_n.\* $OUT_DIR
