#!/bin/bash

#this script computes the difference, one by one, between all files with extension $1 in folder $2 and all  files with extension $2 in folder $3
# run with ./diff-one-by-one.sh txt A B 

clear
clear

DIR_A=$2
DIR_B=$3
EXTENSION=$1

find $DIR_A/ -name '*.'$EXTENSION -exec bash -c 'diff "$1" "'$DIR_B'/${1#'$DIR_A'/}"' _ {} \;