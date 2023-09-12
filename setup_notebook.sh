#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=rtx8000-short
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 0-4:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
ssh_port=2234

# print tunneling instructions jupyter-log
echo -e "
For more info and how to connect from windows,
   see https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/

MacOS or linux terminal command to create your ssh tunnel
ssh -L ${port}:${node}:${port} -N ${user}@oasis.ariseus.net -p${ssh_port}

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: oasis.ariseus.net
SSH login: ${user}
SSH port: ${ssh_port}

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# uncomment the following two lines to use your conda environment called notebook_env
# module load miniconda
# source activate notebook_env

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
jupyter-notebook --no-browser --port="${port}" --ip="${node}"
