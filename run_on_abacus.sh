#!/bin/bash

# run with
# ./run_on_abacus.sh [path where the execution code is located] [name of the folder and job on abacus] [path of mesh solution] [name of problem to solve]
# ./run_on_abacus.sh dynamics/lagrangian_approach/one_dimension/line line_1 ~/Documents/finite_elements/generate_mesh/1d/line/solution line_a

clear
clear

OUT=mcastel1@abacus

echo "Problem: $4"



#clean up 
# move $2 to trash and then launch clean_up which will delete trash
ssh $OUT "mv "$2" trash/'$2'_"$(date +%d_%m_%Y_%H_%M_%S)"; cd clean_up/; sbatch clean_up.slurm"

#create brand-new folder
ssh $OUT "mkdir -p "$2"/mesh"

# replace FOLDER_NAME into script_slurm_abacus.slurm with the actual name of the folder where the job will be executed
rm script_slurm_abacus.slurm
sed 's/FOLDER_NAME/'$2'/; s/PROBLEM_NAME/'$4'/' script_slurm_abacus_blank.slurm > script_slurm_abacus.slurm


# copy mesh, code and modules to abacus
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
rsync -av $3/../mesh_parameters.csv "$OUT:$2/mesh"

# submit the job
ssh $OUT "cd "$2"; sbatch script_slurm_abacus.slurm"

# clean up
rm script_slurm_abacus.slurm