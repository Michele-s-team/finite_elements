#!/bin/bash
#SBATCH --job-name=channel_with_cylinder_flat_icps
#SBATCH --ntasks=1
#SBATCH --mem=128gb
#SBATCH --time=00-01:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."


RUN="/mnt/beegfs/home/mcastel1/channel_with_cylinder_flat_icps_1"
MESH=$RUN"/mesh/solution"
SOLUTION=$RUN"/solution"
PROBLEM="box_ball"

rm -rf $SOLUTION
mkdir $SOLUTION

cd $RUN
#res = 0.1
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 solve.py $PROBLEM $MESH $SOLUTION 
