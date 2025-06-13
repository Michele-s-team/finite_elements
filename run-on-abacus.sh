#!/bin/bash

# run with ./run-on-abacus.sh steady-state-no-flow steady_state_no_flow_1 ~/Documents/finite_elements/generate_mesh/2d/ring/symmetric/solution

clear
clear

OUT=mcastel1@abacus

ssh $OUT "rm -rf "$2
ssh $OUT "mkdir -p "$2"/mesh"

rsync -av modules/*.py $OUT:$2
rsync -av $1/*.py $OUT:$2
rsync -av /Users/michelecastellana/Documents/finite_elements/script_slurm_abacus.slurm $OUT:$2
rsync -av --exclude 'mesh.msh' --exclude '*.csv' --exclude '.DS_Store' $3 $OUT:$2/mesh

ssh $OUT "cd "$2"; sbatch script_slurm_abacus.slurm"