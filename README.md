<div><h2>[CVPR'25] U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion
Framework with Details Enhancement for PAN-Sharpening </h2></div>
<br>

**[Sungpyo Kim](https://www.viclab.kaist.ac.kr/)<sup>1</sup>, [Jeonghyeok Do](https://sites.google.com/view/jeonghyeokdo)<sup>1</sup>, [Jaehyup Lee](https://sites.google.com/view/knuairlab/)<sup>2†</sup>, [Munchurl Kim](https://www.viclab.kaist.ac.kr/)<sup>1†</sup>** 
<br>
<sup>1</sup>KAIST, South Korea, <sup>2</sup>Kyungpook National University, South Korea
<br>
†Co-corresponding authors
<p align="center">
        <a href="https://kaist-viclab.github.io/U-Know-DiffPAN-site/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Kim_U-Know-DiffPAN_An_Uncertainty-aware_Knowledge_Distillation_Diffusion_Framework_with_Details_Enhancement_CVPR_2025_paper.pdf" target='_blank'>
        <img src="https://img.shields.io/badge/2025-CVPR Paper-brightgreen">
        <!-- </a>
        <a href="https://arxiv.org/abs/2412.09982" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.13528-b31b1b.svg"> -->
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/U-Know-DiffPAN">
</p>

<p align="center" width="100%">
    <!-- <img src="https://github.com/KAIST-VICLab/SplineGS/blob/main/assets/architecture.png?raw=tru">  -->
    <img src="https://github.com/KAIST-VICLab/U-Know-DiffPAN/blob/main/assets/U-KnowDiffPAN_figure.png?raw=tru"> 
</p>

## 📣 News
### Updates
- **July 2, 2025**: Code released.
- **February 26, 2025**: U-Know-DiffPAN accepted to CVPR 2025 🎉.
<!-- - **December 13, 2024**: Paper uploaded to arXiv. Check out the manuscript [here](https://arxiv.org/abs/2412.09982).(https://arxiv.org/abs/2412.09982). -->
### To-Dos
- Add instructions for environmental setups.
- Add instructions for Data preperation setups.
- Add instructions for Train and Test.

## ⚙️ Environmental Setups
Clone the repo and install dependencies:
```sh
git clone https://github.com/KAIST-VICLab/U-Know-DiffPAN.git
cd U-Know-DiffPAN

TBA ...
```
<!-- # install splinegs environment
conda create -n splinegs python=3.7 
conda activate splinegs
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install nvidia/label/cuda-11.7.0::cuda
conda install nvidia/label/cuda-11.7.0::cuda-nvcc
conda install nvidia/label/cuda-11.7.0::cuda-runtime
conda install nvidia/label/cuda-11.7.0::cuda-cudart


pip install -e submodules/simple-knn
pip install -e submodules/co-tracker
pip install -r requirements.txt

# install depth environment
conda deactivate
conda create -n unidepth_splinegs python=3.10
conda activate unidepth_splinegs

pip install -r requirements_unidepth.txt
conda install -c conda-forge ld_impl_linux-64
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda install nvidia/label/cuda-12.1.0::cuda
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
conda install nvidia/label/cuda-12.1.0::cuda-runtime
conda install nvidia/label/cuda-12.1.0::cuda-cudart
conda install nvidia/label/cuda-12.1.0::libcusparse
conda install nvidia/label/cuda-12.1.0::libcublas
cd submodules/UniDepth/unidepth/ops/knn;bash compile.sh;cd ../../../../../
cd submodules/UniDepth/unidepth/ops/extract_patches;bash compile.sh;cd ../../../../../

pip install -e submodules/UniDepth
mkdir -p submodules/mega-sam/Depth-Anything/checkpoints -->


## 📁 Data Preparations
### Pancollection Dataset
1. We follow the evaluation setup from [Pancollection](https://github.com/liangjiandeng/PanCollection). Download the datasets [here](https://github.com/liangjiandeng/PanCollection) and arrange them as follows:
```bash
Pancollection
    ├── training_data
    │   ├── train_wv3.h5
    │   ├── train_qb.h5
    │   └── train_gf2.h5
    │
    ├── test_data
    │   ├── test_wv3_OrigScale_multiExm1.h5
    │   ├── test_wv3_multiExm1.h5
    │   └── ...
    │
    └── ...
```

## 🚀 Get Started
## Training
Before starting training, make sure to modify the configuration files (Prior_config.py, FSA_T_config.py, FSA_S_config.py) to match your local environment (e.g., dataset paths, checkpoint directories, device settings, etc.).
```sh
# check if environment is activated properly
conda activate uknow
```
### Priornet pretrain
```sh
python main.py --stage Priornet --mode train
```
### Stage 1
```sh
# FSA-T Pretrain
python main.py --stage FSA_T --mode train

# Get FSA-T output from train datasets  
python main.py --stage FSA_T --mode save
```

After getting the .h5 format FSA-T output (dist_feature_map), arrange datasetes as follows:
```bash
Pancollection
    ├── training_data
    │   └── ...
    │
    ├── test_data
    │   └── ...
    │
    ├── dist_feature_map
    │   ├── train_wv3.h5
    │   ├── train_qb.h5
    │   └── train_gf2.h5
    │
    └── ...
```

### Stage 2
```sh
# FSA-S Train
python main.py --stage FSA_S --mode train
```
###  Test
```sh
# FSA-S Test
python main.py --stage FSA_S --mode test
```
<!-- #### Metrics Evaluation
```sh
python eval_nvidia.py -s data/nvidia_rodynrf/${SCENE}/ --expname "${EXP_NAME}" --configs arguments/nvidia_rodynrf/${SCENE}.py --checkpoint output/${EXP_NAME}/point_cloud/fine_best
``` -->
<!-- #### Training
T.B.D -->
### Evaluation
TBA ...

## Acknowledgments
This work was supported by National Research Foundation of Korea (NRF) grant funded by the Korean Government
[Ministry of Science and ICT (Information and Communications Technology)] (Project Number: RS- 2024-00338513, Project Title: AI-based Computer Vision Study for Satellite Image Processing and Analysis, 100%).

## ⭐ Citing U-Know-DiffPAN

If you find our repository useful, please consider giving it a star ⭐ and citing our research papers in your work:
```bibtex
@inproceedings{kim2025u,
  title={U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for PAN-Sharpening},
  author={Kim, Sungpyo and Do, Jeonghyeok and Lee, Jaehyup and Kim, Munchurl},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={23069--23079},
  year={2025}
}
```

