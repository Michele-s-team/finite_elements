#!/bin/bash
#SBATCH --job-name=steady_state_no_flow_1
#SBATCH --ntasks=1
#SBATCH --mem=128gb
#SBATCH --time=00-04:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."

#CHANGE PARAMETERS HERE
# T=0.1
# k=1.0 
# rho=1.0 
# eta=1.0
# v=1.0 
# N=1048576
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/steady_state_no_flow_1"
MESH=$RUN"/mesh/solution"
SOLUTION=$RUN"/solution"
PROBLEM="ring"

rm -rf $SOLUTION
mkdir $SOLUTION

cd $RUN
#res = 0.0.00625
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $PROBLEM $MESH $SOLUTION 
