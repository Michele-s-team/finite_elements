#!/bin/bash
#SBATCH --job-name=FOLDER_NAME
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=04-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."

RUN="/mnt/beegfs/home/mcastel1/FOLDER_NAME"
MESH=$RUN"/mesh/solution"
SOLUTION=$RUN"/solution"
PROBLEM="PROBLEM_NAME"

rm -rf $SOLUTION
mkdir $SOLUTION

cd $RUN
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $PROBLEM $MESH $SOLUTION 
