#!/bin/bash

clear
clear

OUT=mcastel1@abacus

ssh $OUT "rm -rf "$2"/*.py "
ssh $OUT "rm -rf "$2"/solution"
ssh $OUT "mkdir "$2"/mesh"


scp modules/*.py $OUT:$2
scp $1/*.py $OUT:$2

rsync -av --exclude 'mesh.msh' --exclude '.DS_Store' $1/mesh $OUT:$2/mesh
rsync -av $1/*.py $OUT:$2
