#!/bin/bash

conda create -y -n stablediff_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
source activate stablediff_env
export PYTHONNOUSERSITE=True
pip install torch 
pip install transformers ftfy scipy accelerate
pip install diffusers==0.13.0
pip install opencv-python-headless==4.7.0.68
pip install xformers
pip install conda-pack
conda-pack
