#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --job-name=300KCluster
#SBATCH --account=project_2001083
#SBATCH --mem=128000M
#SBATCH --output=out/A300K_Cluster.log

module load tykky
export PATH="/scratch/project_2001083/nasib/Environments/xc/bin:$PATH"

python src/cluster.py