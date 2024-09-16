#!/bin/bash
#SBATCH --job-name=fenics_example
#SBATCH --ntasks=1
##SBATCH --mem=1gb
#SBATCH --time=10-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Cantami o diva ... l'ira di Achille ... "

IN="/mnt/beegfs/home/mcastel1/fenics_example"
RUN="/mnt/beegfs/home/mcastel1/fenics_example"
OUT=$RUN"/solution/"
T="0.001"
N="3"

rm -rf $OUT
mkdir $OUT

cd $RUN
apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 example.py $IN $OUT $T $N
