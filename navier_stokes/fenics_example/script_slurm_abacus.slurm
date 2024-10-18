#!/bin/bash
#SBATCH --job-name=dynamics-bcs-w_T0.1_kappa1_rho1_eta1_N16384_res0.1
#SBATCH --ntasks=1
#SBATCH --mem=128gb
#SBATCH --time=128-00:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Arma virumque  cano , Troiae qui primus ab oris Italiam, fato profugus, Laviniaque venit litora, multum ille et terris iactatus et alto"

#CHANGE PARAMETERS HERE
T=0.1
N=16384
kappa=1.0
rho=1.0
eta=1.0
res=0.1
#CHANGE PARAMETERS HERE

RUN="/mnt/beegfs/home/mcastel1/dynamics-bcs-w_T"$T"_kappa"$kappa"_rho"$rho"_eta"$eta"_N"$N"_res"$res
MESH=$RUN
SOLUTION=$RUN"/solution"

rm -rf $SOLUTION
mkdir $SOLUTION

cd $RUN
#python3 generate_mesh.py 0.1
time apptainer exec  /mnt/beegfs/common/containers/singularity/dev/FEniCS/FEniCS.sif python3 example.py $MESH $SOLUTION $T $kappa $rho $eta $N
