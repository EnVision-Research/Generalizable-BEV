# Generalizable-BEV
<div style="text-align:center;">
  <img src="/Framework.png" style="width:85%;" />
</div>



## Get Started

#### Installation and Data Preparation

step 1. Prepare the environment refer to [BEVDet](https://github.com/HuangJunJie2017/BEVDet).

step 2. Prepare PDBEV repo by.
```shell script
git clone https://github.com/EnVision-Research/Generalizable-BEV.git
cd Generalizable-BEV
pip install -v -e .
```

step 3. Prepare datasets:

nuScenes: [Details](https://github.com/HuangJunJie2017/BEVDet)

DeepAccident: [Details](https://github.com/tianqi-wang1996/DeepAccident)

Lyft: [Details](https://github.com/lyft/nuscenes-devkit)


The pre-processed pkl of the three data sets can be downloaded directly [here].

step 4. Train for domain generealization:

```
bash tools/dist_train.sh  ./configs/PDBEV/pdbev-r50-cbgs-NUS2X-dg.py   8     # nuScenes as source domain, using 8 gpus
bash tools/dist_train.sh  ./configs/PDBEV/pdbev-r50-cbgs-LYFT2X-dg.py  8     # Lyft as source domain, using 8 gpus
bash tools/dist_train.sh  ./configs/PDBEV/pdbev-r50-cbgs-DA2X-dg.py    8     # DeepAccident as source domain, using 8 gpus
```

step 5. Train for unspuervised domain adapataion:
```
bash tools/dist_train.sh  ./configs/PDBEV/pcbev-uda-NUS2LYFT.py  8 --checkpoint ./work_dirs/pdbev-r50-cbgs-NUS2X-dg/epoch_23.pth
# nuScenes as source domain, LYFT as target domain, using 8 gpus, loading DG pretrain models at 23 epoch
# You only need to modify the path of the configuration file of different data set D and the corresponding model M to test the performance of model M on the corresponding data set D. It is worth mentioning that none of our algorithms change the model infrastructure, so they are only used for BEVDepth evaluation.
```

step 6. Test at target domain:
```
bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-lyft.py  ./work_dirs/pdbev-r50-cbgs-NUS2X-dg/epoch_24.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-nus/nus.pkl
# nuScenes as source domain, tested on LYFT target domain, loading DG pretrain models at 24 epoch
# You only need to modify the path of the configuration file of different data set D and the corresponding model M to test the performance of model M on the corresponding data set D. It is worth mentioning that none of our algorithms change the model infrastructure, so they are only used for BEVDepth evaluation.
```

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [DeepAccident](https://github.com/tianqi-wang1996/DeepAccident)
- [Lyft](https://github.com/lyft/nuscenes-devkit)


## Citation
```
@InProceedings{PD-BEV,
    author    = {Hao LU, Yunpeng ZHANG, Qing LIAN, Dalong DU, Ying-Cong CHEN},
    title     = {Towards Generalizable Multi-Camera 3D Object Detection via Perspective Debiasing},
    booktitle = {arXiv preprint arXiv:2310.11346},
    year      = {2023},
}
```
