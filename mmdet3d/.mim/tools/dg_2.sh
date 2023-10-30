cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate DA2
cd /mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed

# bash tools/dist_train.sh /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/configs/1_DA_base_pretrain/bevdepth-r50-cbgs-DA2X-dg-base.py 8

#bash tools/dist_train.sh ./configs/111_UDA_pretrain_box/bevbox-r50-cbgs-DA2X-dg.py 8 # --checkpoint /mnt/cfs/algorithm/hao.lu/Code/BEVDepth_DA/work_dirs/bevdepth-r50-cbgs-DA2X-dg-base/epoch_10.pth




#bash tools/dist_train.sh  ./configs/111111_PCBEV/pcbev-r50-cbgs-NUS2X-dg.py  8


bash tools/dist_train.sh  ./configs/bevdet_our/bevdepth-r50-cbgs-pc-nus.py  8
