#!/bin/bash

# run with ./copy_to_abacus.sh steady-state-flow steady-state-flow

clear
clear

OUT=mcastel1@abacus

ssh $OUT "rm -rf "$2"/*.py "
ssh $OUT "rm -rf "$2"/solution"
ssh $OUT "mkdir "$2"/mesh"


scp modules/*.py $OUT:$2
scp $1/*.py $OUT:$2

rsync -av --exclude 'mesh.msh' --exclude '.DS_Store' $1/mesh $OUT:$2
rsync -av $1/*.py $OUT:$2
