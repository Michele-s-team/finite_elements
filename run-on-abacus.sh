#!/bin/bash

# run with ./run-on-abacus.sh steady-state-flow steady-state-flow
# run with ./run-on-abacus.sh dynamics dynamics

clear
clear

OUT=mcastel1@abacus

ssh $OUT "rm -rf "$2
ssh $OUT "mkdir -p "$2"/mesh"
ssh $OUT "mkdir -p "$2"/solution/snapshots/csv"


rsync -av modules/*.py $OUT:$2
rsync -av $1/*.py $OUT:$2
rsync -av /Users/michelecastellana/Documents/finite_elements/script_slurm_abacus.slurm $OUT:$2
rsync -av --exclude 'mesh.msh' --exclude '.DS_Store' $1/mesh $OUT:$2

ssh $OUT "cd "$2"; sbatch script_slurm_abacus.slurm"