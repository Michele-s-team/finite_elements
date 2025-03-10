#!/bin/bash
#SBATCH --job-name=dynamics-3
#SBATCH --ntasks=1
#SBATCH --mem=128gb
#SBATCH --time=128-00:00:00
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
N=1048576
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/dynamics-3"
MESH=$RUN"/mesh"
SOLUTION=$RUN"/solution"

rm -rf $SOLUTION
mkdir $SOLUTION
mkdir -p $SOLUTION"/snapshots/csv/"

cd $RUN
#res = 0.1
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $MESH $SOLUTION $T $k $rho $eta $v $N
