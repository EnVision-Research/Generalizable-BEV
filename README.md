# Generalizable-BEV [AAAI 2025]
<div style="text-align:center;">
  <img src="/Framework.png" style="width:95%;" />
</div>

A plug-and-play BEV generalization framework that can leverage both unlabeled and labeled data.

## Get Started


#### **step 1. Prepare the environment refer to [BEVDet](https://github.com/HuangJunJie2017/BEVDet).**

#### **step 2. Prepare PDBEV repo by.**
```shell script
git clone https://github.com/EnVision-Research/Generalizable-BEV.git
cd Generalizable-BEV
pip install -v -e .
```

#### **step 3. Prepare datasets:**
The preparation of the dataset is actually to generate the corresponding index (pkl files), which can then be used with the dataset that we have created.

###### nuScenes dataset pkl file generation refers to [Details](https://github.com/HuangJunJie2017/BEVDet)

###### DeepAccident dataset pkl file generation refers to [Details](https://github.com/tianqi-wang1996/DeepAccident), and then use ./tools/Deepaccident_converter.py to convert to a uniform format.

###### Lyft use ./tools/Lyft_converter.py to convert to a uniform format.

The pre-processed pkl of the three data sets can be downloaded directly [here].

#### **step 4. Train for domain generalization:**
```
bash tools/dist_train.sh  $confige_file$  $Gpus_num$     
```
For example:
```
bash tools/dist_train.sh  ./configs/PDBEV/pdbev-r50-cbgs-NUS2X-dg.py   8     # nuScenes as source domain, using 8 gpus
bash tools/dist_train.sh  ./configs/PDBEV/pdbev-r50-cbgs-LYFT2X-dg.py  8     # Lyft as source domain, using 8 gpus
bash tools/dist_train.sh  ./configs/PDBEV/pdbev-r50-cbgs-DA2X-dg.py    8     # DeepAccident as source domain, using 8 gpus
```

#### **step 5. Train for unsupervised domain adaptation:**
```
bash tools/dist_train.sh  $confige_file$  c   --checkpoint  $the pretrained models on source domain$
```
For example:
```
bash tools/dist_train.sh  ./configs/PDBEV/pcbev-uda-NUS2LYFT.py  8 --checkpoint ./work_dirs/pdbev-r50-cbgs-NUS2X-dg/epoch_23.pth
# nuScenes as source domain, LYFT as target domain, using 8 gpus, loading DG pretrain models at 23 epoch
# You only need to modify the path of the configuration file of different data set D and the corresponding model M to test the performance of model M on the corresponding data set D. It is worth mentioning that none of our algorithms change the model infrastructure, so they are only used for BEVDepth evaluation.
```

#### **step 6. Test at target domain:**
```
bash ./tools/dist_test.sh &test dataset config_file&  &model_path&   $Gpus_num$  --eval bbox --out  $output_path$
```
For example:
```
bash ./tools/dist_test.sh ./configs/bevdet_our/bevdepth-r50-cbgs-pc-lyft.py  ./work_dirs/pdbev-r50-cbgs-NUS2X-dg/epoch_24.pth 8 --eval bbox --out ./work_dirs/bevdepth-r50-cbgs-pc-nus/nus.pkl
```

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples: [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [DeepAccident](https://github.com/tianqi-wang1996/DeepAccident), [Lyft](https://github.com/lyft/nuscenes-devkit).

## Email

hlu585@connect.hkust-gz.edu.cn


## Citation
```
@InProceedings{lu2025PD-BEV,
    author    = {Hao LU, Yunpeng ZHANG, Guoqing WANG, Qing LIAN, Dalong DU, Ying-Cong CHEN},
    title     = {Towards Generalizable Multi-Camera 3D Object Detection via Perspective Debiasing},
    booktitle = {AAAI},
    year      = {2025},
}
```
