#!/bin/bash

conda create -y -n lama_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
source activate lama_env
export PYTHONNOUSERSITE=True

pip install torch 
pip install opencv-python
pip install pyyaml
pip install tqdm
pip install numpy==1.23
pip install easydict==1.9.0
pip install scikit-image==0.17.2
pip install scikit-learn==0.24.2
pip install tensorflow
pip install joblib
pip install matplotlib
pip install pandas
pip install albumentations==0.5.2
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install tabulate
pip install kornia==0.5.0
pip install webdataset
pip install packaging
pip install omegaconf==2.1.2
pip install torchvision
pip install conda-pack
conda-pack

