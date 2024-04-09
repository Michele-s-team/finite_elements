#!/bin/bash

IN_DIR="/mnt/beegfs/home/mcastel1/navier_stokes/results"
OUT_DIR="/Users/michele/Desktop"

rsync --size-only -P -v -e ssh mcastel1@abacus:$IN_DIR/velocity.\* $OUT_DIR
