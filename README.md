# NetworkGenerator
ENV_NAME=discrete_guidance; conda create -n $ENV_NAME --yes python=3.9
conda activate $ENV_NAME; ./install.sh $ENV_NAME
Download random.npz to the path 'GS_project/data'
python GS_project/discrete_guidance/data_transfer.py
