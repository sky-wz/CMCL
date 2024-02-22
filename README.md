# Contrastive Multi-bit Collaborative Learning for Deep Cross-modal Hashing


### 1. Introduction

This is the source code of paper "Contrastive Multi-bit Collaborative Learning for Deep Cross-modal Hashing".


### 2. Requirements

- python 3.7.6
- pytorch 1.8.0
- torchvision 0.9.0
- numpy
- scipy
- tqdm
- pillow
- einops
- ftfy
- regex
- ...




### 3. Preparation

#### 3.1 Download pre-trained CLIP

Pretrained CLIP model could be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). 
This code is based on the "ViT-B/32". 
You should download "ViT-B/32" and put it in `./cache`, or you can find it from the following link:

link: https://pan.baidu.com/s/1ZyDTR2IEHlY4xIdLgxtaVA 
password: kdq7


#### 3.2 Generate dataset

You should generate the following `*.mat` file for each dataset. The structure of directory `./dataset` should be:
```
    dataset
    ├── coco
    │   ├── caption.mat 
    │   ├── index.mat
    │   └── label.mat 
    ├── flickr25k
    │   ├── caption.mat
    │   ├── index.mat
    │   └── label.mat
    └── nuswide
        ├── caption.mat
        ├── index.mat 
        └── label.mat
```

Please preprocessing dataset to appropriate input format.

More details about the generation, meaning, and format of each mat file could be found in `./dataset/README.md`.

Additionally, cleaned datasets (MIRFLICKR25K & MSCOCO & NUSWIDE) used in our experiments are available at `pan.baidu.com`:

link: https://pan.baidu.com/s/1ZyDTR2IEHlY4xIdLgxtaVA 
password: kdq7


### 4. Train

After preparing the python environment, pretrained CLIP model, and dataset, we can train the CMCL model.

See `CMCL_$DATASET$.sh`.

### 5. Test
See `CMCL_$DATASET$_TEST.sh`.
