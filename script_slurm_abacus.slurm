#!/bin/bash
#SBATCH --job-name=dynamics-1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=04-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."

#CHANGE PARAMETERS HERE
T=0.1
kappa=1e-06
rho=0.01293
eta=1.85e-07
v=1
N=100000
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/dynamics-1"
MESH=$RUN"/mesh/solution"
SOLUTION=$RUN"/solution"

rm -rf $SOLUTION
mkdir $SOLUTION
mkdir -p $SOLUTION"/snapshots/csv/nodal_values/"

cd $RUN
#res = 0.1
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $MESH $SOLUTION $T $kappa $rho $eta $v $N
