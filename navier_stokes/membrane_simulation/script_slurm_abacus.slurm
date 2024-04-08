#!/bin/bash
#SBATCH --job-name=navier_stokes
#SBATCH --ntasks=1
##SBATCH --mem=1gb
#SBATCH --time=10-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Cantami o diva ... l'ira di Achille ... "

RUN="/mnt/beegfs/home/mcastel1/navier_stokes/"
OUT=$RUN"/results/"

rm -rf $OUT
mkdir $OUT

cd $RUN
apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 navier_stokes_membrane.py
