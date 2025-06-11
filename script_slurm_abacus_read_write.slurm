#!/bin/bash
#SBATCH --job-name=read_write_1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=00-08:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."


RUN="/mnt/beegfs/home/mcastel1/read_write_1"
PATH_IN="/mnt/beegfs/home/mcastel1/dynamics_1"
SOLUTION_IN=$PATH_IN"/solution/snapshots/h5"
MESH=$PATH_IN"/mesh/solution"
SOLUTION_OUT=$RUN"/solution"
PROBLEM="square"
INCREMENT=10

rm -rf $SOLUTION_OUT
mkdir $SOLUTION_OUT

cd $RUN
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $PROBLEM $MESH $SOLUTION_IN $SOLUTION_OUT $INCREMENT 
