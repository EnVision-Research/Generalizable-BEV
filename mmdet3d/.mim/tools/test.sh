cd /mnt/cfs/algorithm/hao.lu/
source /mnt/cfs/algorithm/hao.lu/temp/.bashrc
conda activate DA2
cd /mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed


bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-nus.py  ./work_dirs/pcbev-r50-cbgs-NUS2X-dg/epoch_23.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-nus/nus.pkl
bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-nus.py  ./work_dirs/pcbev-r50-cbgs-LYFT2X-dg/epoch_23.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-nus/lyft2nus.pkl
bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-nus.py  ./work_dirs/pcbev-r50-cbgs-DA2X-dg/epoch_3.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-nus/da2nus.pkl

#bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-lyft.py  ./work_dirs/pcbev-r50-cbgs-LYFT2X-dg/epoch_23.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-lyft/lyft2lyft.pkl
#bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-lyft.py  ./work_dirs/pcbev-r50-cbgs-NUS2X-dg/epoch_23.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-lyft/nus2lyft.pkl
#bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-lyft.py  ./work_dirs/pcbev-r50-cbgs-DA2X-dg/epoch_3.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-lyft/da2lyft.pkl


#bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-da.py  ./work_dirs/pcbev-r50-cbgs-DA2X-dg/epoch_4.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-da/da2da.pkl
#

#bash tools/dist_train.sh  ./configs/111111_PCBEV/pcbev-uda-DA2LYFT.py  8  --checkpoint /mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed/work_dirs/pcbev-r50-cbgs-DA2X-dg/epoch_3.pth
#
#bash tools/dist_train.sh  ./configs/111111_PCBEV/pcbev-uda-DA2NUS.py   8 --checkpoint /mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed/work_dirs/pcbev-r50-cbgs-DA2X-dg/epoch_3.pth
#
#bash tools/dist_train.sh  ./configs/111111_PCBEV/pcbev-uda-NUS2LYFT.py   8 --checkpoint /mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed/work_dirs/pcbev-r50-cbgs-NUS2X-dg/epoch_23.pth
#
#bash tools/dist_train.sh  ./configs/111111_PCBEV/pcbev-uda-LYFT2NUS.py   8 --checkpoint /mnt/cfs/algorithm/hao.lu/Code/PCBEV_realsed/work_dirs/pcbev-r50-cbgs-LYFT2X-dg/epoch_23.pth
#
#
#
