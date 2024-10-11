#!/bin/bash
#SBATCH --job-name=test_mpi
#SBATCH --ntasks=2
#SBATCH --mem=128gb
#SBATCH --time=10-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Arma virumque  cano , Troiae qui primus ab oris Italiam, fato profugus, Laviniaque venit litora, multum ille et terris iactatus et alto"

#CHANGE PARAMETERS HERE
T=0.1
N=1024
r=0.3
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/test_mpi"
MESH=$RUN
SOLUTION=$RUN"/solution"

rm -rf $SOLUTION
mkdir $SOLUTION

cd $RUN
#python3 generate_mesh.py 0.3
time mpiexec -np 2 apptainer exec /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif  python3 example.py $MESH $SOLUTION $T $N
