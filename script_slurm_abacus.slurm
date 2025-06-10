#!/bin/bash
#SBATCH --job-name=fluid_structure_interaction
#SBATCH --ntasks=1
#SBATCH --mem=128gb
#SBATCH --time=01-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."


RUN="/mnt/beegfs/home/mcastel1/fluid_structure_interaction_1"
MESH=$RUN"/mesh/solution"
SOLUTION=$RUN"/solution"
PROBLEM="square_ellipse"

rm -rf $SOLUTION
mkdir $SOLUTION

cd $RUN
#res = 0.05
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $PROBLEM $MESH $SOLUTION 
