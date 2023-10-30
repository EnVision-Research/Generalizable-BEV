cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate DA1
cd /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA
export PYTHONPATH="."


config=$1
python tools/launch_dist_job.py $config /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA