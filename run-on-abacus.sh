#!/bin/bash

# run with ./run-on-abacus.sh read_write read_write_1 ~/Documents/finite_elements/generate_mesh/2d/square/solution

clear
clear

OUT=mcastel1@abacus

# CHANGE PARAMETERS HERE
# SCRIPT_SLURM="script_slurm_abacus.slurm"
SCRIPT_SLURM="script_slurm_abacus_read_write.slurm"
# CHANGE PARAMETERS HERE

ssh $OUT "rm -rf "$2
ssh $OUT "mkdir -p "$2"/mesh"

rsync -av modules/*.py $OUT:$2
rsync -av $1/*.py $OUT:$2
rsync -av /Users/michelecastellana/Documents/finite_elements/$SCRIPT_SLURM $OUT:$2
rsync -av --exclude 'mesh.msh' --exclude '*.csv' --exclude '.DS_Store' $3 $OUT:$2/mesh

ssh $OUT "cd "$2"; sbatch $SCRIPT_SLURM"