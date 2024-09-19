#!/bin/bash
#SBATCH --job-name=fenics_example_T0.1_N128_r0.1
#SBATCH --ntasks=1
##SBATCH --mem=1gb
#SBATCH --time=10-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Cantami o diva ... l'ira di Achille ... "

#CHANGE PARAMETERS HERE
r="0.1"
T="0.1"
N="128"
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/fenics_example_T"$T"_N"$N"_r"$r
IN=$RUN
OUT=$RUN"/solution/"


rm -rf $OUT
mkdir $OUT

cd $RUN
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 example.py $IN $OUT $T $N
