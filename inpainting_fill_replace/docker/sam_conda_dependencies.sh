#!/bin/bash

conda create -y -n sam_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
source activate sam_env
export PYTHONNOUSERSITE=True
pip install torch
pip install segment-anything-py==1.0
pip install opencv-python-headless==4.7.0.68
pip install matplotlib==3.6.3
pip install conda-pack
conda-pack
