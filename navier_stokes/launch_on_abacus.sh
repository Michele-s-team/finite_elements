#!/bin/bash

#rsync --size-only -P -v -e ssh mcastel1@abacus:/mnt/beegfs/home/mcastel1/navier_stokes/results/velocity.\* ~/Desktop


CODE_DIR="/Users/michele/Documents/fenics/navier_stokes/membrane_simulation"
MESH_DIR="/Users/michele/Documents/fenics/mesh/membrane_mesh"
DEST_DIR="/mnt/beegfs/home/mcastel1/navier_stokes"

ssh mcastel1@abacus "rm -rf "$DEST_DIR"/*"

scp $CODE_DIR/*.py $CODE_DIR/script_slurm_abacus.slurm mcastel1@abacus:$DEST_DIR
scp $MESH_DIR/*.h5 $MESH_DIR/*.xdmf $MESH_DIR/*.msh mcastel1@abacus:$DEST_DIR

ssh mcastel1@abacus "cd "$DEST_DIR" ; sbatch script_slurm_abacus.slurm"
