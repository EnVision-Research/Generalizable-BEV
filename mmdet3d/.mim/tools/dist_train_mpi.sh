#!/bin/bash
cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate DA1
cd /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA


CONFIG=$1
nproc_per_node=$2
master_addr=$3
nnodes=$4
node_rank=$5
PORT=${PORT:-29506}

export OMP_NUM_THREADS=8
export PYTHONPATH="$(dirname $0)/../":$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:6}