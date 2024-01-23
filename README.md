# Deep learning based Image Compression for Microscopy Images

## Introduction
This repo is the code base for the paper: **Deep learning based Image Compression for Microscopy Images: An Empirical Study**

The submodule `CompressAI` is forked from [CompressAI](https://github.com/InterDigitalInc/CompressAI) and adapted to the grayscale and 2/3D microscopy images.

## Installation
- clone the repo:
    ```
    git clone --recurse-submodules https://github.com/MMV-Lab/data-compression.git
    cd CompressAI
    ```
- install the environment (requires conda):
  ```
  conda create -n CompressAI python=3.9
  conda activate CompressAI
  pip install -e .    
  ```
- also need to install the **mmv_im2im package** if you want to do the downstream labelfree task. Please check the [github repo](https://github.com/MMV-Lab/mmv_im2im) for the installation.
## Reproducibility

You can go to [this folder](paper/paper_exp) and try out our jupyter notebooks for both 2D and 3D tasks.

We will release the pretrained models and dataset to Zenodo.
## Acknowledgement
This project is the application and adaptation of the [CompressAI](https://github.com/InterDigitalInc/CompressAI) tool in the bioimage field.
## Citation
```
@article{zhou2023deep,
  title={Deep learning based Image Compression for Microscopy Images: An Empirical Study},
  author={Zhou, Yu and Sollmann, Jan and Chen, Jianxu},
  journal={arXiv preprint arXiv:2311.01352},
  year={2023}
}
```