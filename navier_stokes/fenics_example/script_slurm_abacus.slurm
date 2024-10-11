#!/bin/bash
#SBATCH --job-name=name_of_the_job
#SBATCH --ntasks=128
#SBATCH --mem=1gb
#SBATCH --time=00-01:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Arma virumque  cano , Troiae qui primus ab oris Italiam, fato profugus, Laviniaque venit litora, multum ille et terris iactatus et alto"

#CHANGE PARAMETERS HERE
T=0.1
N=16
r=0.1
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/name_of_the_folder"
MESH=$RUN
SOLUTION=$RUN"/solution"

rm -rf $SOLUTION
mkdir $SOLUTION

cd $RUN
#python3 generate_mesh.py 0.1

#run with one core
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 example.py $MESH $SOLUTION $T $N

#to run with multiple cores
#time mpiexec -np 128 apptainer exec /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif  python3 example.py $MESH $SOLUTION $T $N


