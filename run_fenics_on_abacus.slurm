#!/bin/bash
#SBATCH --job-name=test_navier_stokes
#SBATCH --ntasks=1
##SBATCH --mem=1gb
#SBATCH --time=00-00:10:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr


echo "Cantami o diva ..."

RUN="/mnt/beegfs/home/mcastel1/navier_stokes_michele/demo/python/documented/navier-stokes/"

cd $RUN
apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 demo_navier-stokes.py
