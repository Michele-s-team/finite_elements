#!/bin/bash
#SBATCH --job-name=dynamics-4
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=02-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."

#CHANGE PARAMETERS HERE
T=0.1
k=1.0 
rho=1.0 
eta=1.0
v=1.0 
N=65536
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/dynamics-4"
MESH=$RUN"/mesh"
SOLUTION=$RUN"/solution"

rm -rf $SOLUTION
mkdir $SOLUTION
mkdir -p $SOLUTION"/snapshots/csv/"

cd $RUN
#res = 0.1
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $MESH $SOLUTION $T $k $rho $eta $v $N
