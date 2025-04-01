#!/bin/bash
#SBATCH --job-name=dyn-fig-11
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=04-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."

#CHANGE PARAMETERS HERE
T=10
N=1000
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/dynamics-1"
MESH=$RUN"/mesh"
SOLUTION=$RUN"/solution"

rm -rf $SOLUTION
mkdir $SOLUTION
mkdir -p $SOLUTION"/snapshots/csv/"

cd $RUN
#res = 1
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $MESH $SOLUTION $T $N
