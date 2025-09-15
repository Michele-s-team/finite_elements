#!/bin/bash

# run with
# ./run-on-abacus.sh dynamics/lagrangian_approach/one_dimension line_4 ~/Documents/finite_elements/generate_mesh/1d/line/solution

clear
clear

OUT=mcastel1@abacus

ssh $OUT "rm -rf "$2
ssh $OUT "mkdir -p "$2"/mesh"

rsync -av modules/ $OUT:$2
rsync -av $1/*.py $OUT:$2
rsync -av /Users/michelecastellana/Documents/finite_elements/script_slurm_abacus.slurm $OUT:$2
rsync -av \
  --exclude 'mesh.msh' \
  --exclude '.DS_Store' \
  --include 'mesh_metadata.csv' \
  --exclude '*.csv' \
  "$3" "$OUT:$2/mesh"
rsync -av  $1/parameters*.csv $OUT:$2


ssh $OUT "cd "$2"; sbatch script_slurm_abacus.slurm"