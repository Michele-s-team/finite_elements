#!/bin/bash
#SBATCH --job-name=dynamics-1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=01-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."

#CHANGE PARAMETERS HERE
T=1e-4
N=10000
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/dynamics-1"
MESH=$RUN"/mesh/solution"
SOLUTION=$RUN"/solution"

rm -rf $SOLUTION
mkdir $SOLUTION
mkdir -p $SOLUTION"/snapshots/csv/"

cd $RUN
#res = 0.2
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $MESH $SOLUTION $T $N
