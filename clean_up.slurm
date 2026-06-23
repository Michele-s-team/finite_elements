
#!/bin/bash
#SBATCH --job-name=clean_up
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=00-04:00:00
##SBATCH --account=recherche_prod
#SBATCH --partition=dev
#SBATCH --mail-user=michele.castellana@curie.fr

echo "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non interrotte di monti..."

echo "Cleaning up ..."
cd /mnt/beegfs/home/mcastel1/clean_up
rm -rf slurm*
time rm -rf /mnt/beegfs/home/mcastel1/trash/**
echo "... done! :)"
