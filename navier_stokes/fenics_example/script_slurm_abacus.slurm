#!/bin/bash
#SBATCH --job-name=navier_stokes
#SBATCH --ntasks=1
##SBATCH --mem=1gb
#SBATCH --time=10-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Cantami o diva ..."

RUN="/mnt/beegfs/home/mcastel1/navier_stokes/"
OUT=$RUN"/solution/"
IN="/mnt/beegfs/home/mcastel1/navier_stokes/"
T=0.05
N=64

rm -rf $OUT
mkdir $OUT

cd $RUN
apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 example.py $IN $OUT $T $N
