#!/bin/bash

#rsync --size-only -P -v -e ssh mcastel1@abacus:/mnt/beegfs/home/mcastel1/navier_stokes/results/velocity.\* ~/Desktop


CODE_DIR="/Users/michelecastellana/Documents/finite_elements/navier_stokes/fenics_example"
MESH_DIR="/Users/michelecastellana/Documents/finite_elements/mesh/membrane_mesh"

#CHANGE PARAMETERS HERE
T=0.4
N=2048
r=0.3
DEST_DIR="/mnt/beegfs/home/mcastel1/fenics_example_T"$T"_N"$N"_r"$r
#CHANGE PARAMETERS HERE


ssh mcastel1@abacus "rm -rf "$DEST_DIR"/*"

scp $CODE_DIR/*.py $CODE_DIR/script_slurm_abacus.slurm mcastel1@abacus:$DEST_DIR
scp $MESH_DIR/*.h5 $MESH_DIR/*.xdmf $MESH_DIR/*.msh mcastel1@abacus:$DEST_DIR

ssh mcastel1@abacus "cd "$DEST_DIR" ; sbatch script_slurm_abacus.slurm"
