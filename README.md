# Self-Calibration Flow Guided Denoising Diffusion Model for Human Pose Transfer
The source code of our paper: [*"Self-Calibration Flow Guided Denoising Diffusion Model for Human Pose Transfer"*](https://ieeexplore.ieee.org/document/10483084) [Accepted by TCSVT 2024] \
Yu Xue, Lai-Man Po, Wing-Yin Yu, Haoxuan Wu, Xuyuan Xu, Kun Li, Yuyang Liu


## Abstract

The human pose transfer task aims to generate synthetic person images that preserve the style of reference images while accurately aligning them with the desired target pose. However, existing methods based on generative adversarial networks (GANs) struggle to produce realistic details and often face spatial misalignment issues. On the other hand, methods relying on denoising diffusion models require a large number of model parameters, resulting in slower convergence rates. To address these challenges, we propose a self-calibration flow-guided module (SCFM) to establish precise spatial correspondence between reference images and target poses. This module facilitates the denoising diffusion model in predicting the noise at each denoising step more effectively. Additionally, we introduce a multi-scale feature fusing module (MSFF) that enhances the denoising U-Net architecture through a cross-attention mechanism, achieving better performance with a reduced parameter count. Our proposed model outperforms state-of-the-art methods on the DeepFashion and Market-1501 datasets in terms of both the quantity and quality of the synthesized images.


## Generated Results
You can directly download our test set results of the DeepFashion Dataset from [Google Drive](https://drive.google.com/file/d/1B850vIDIN7P2PwpLdFwTjIvZtPtFwA8q/view?usp=sharing).


## Dataset

- We follow the way of processing data like [PIDM](https://github.com/ankanbhunia/PIDM)
- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then rename the obtained folder as **img** and put it under your specified path `specified_path` directory. 

- We split the train/test set following [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention). Several images with significant occlusions are removed from the training set. Download the train/test pairs and the keypoints `pose.zip` extracted with [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) by downloading these files manually：

  - Download the train/test pairs from [Google Drive](https://drive.google.com/drive/folders/1PhnaFNg9zxMZM-ccJAzLIt2iqWFRzXSw?usp=sharing) including **train_pairs.txt**, **test_pairs.txt**, **train.lst**, **test.lst**. Put these files under the `specified_path` directory. 
  - Download the keypoints `pose.rar` extracted with Openpose from [Google Driven](https://drive.google.com/file/d/1waNzq-deGBKATXMU9JzMDWdGsF4YkcW_/view?usp=sharing). Unzip and put the obtained floder under the  `specified_path` directory.

- Run the following code to save images to lmdb dataset.

  ```bash
  python data/prepare_data.py \
  --root specified_path \
  --out specified_path
  ```


## Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n SCFM python=3.6
conda activate SCFM
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 2. Clone the Repo and Install dependencies
git clone https://github.com/zylwithxy/SCFM-guided-DDPM.git
bash setup.sh # install flow module dependencies
pip install -r requirements.txt

```


## Training 

```bash
bash ./scripts/train.sh
```


## Inference 

The trained model is in the ```./checkpoints``` folder. 

```bash
bash ./scripts/inference.sh
```


## Citation

If you use the results and code for your research, please cite our paper:

```
@article{xue2024self,
  title={Self-Calibration Flow Guided Denoising Diffusion Model for Human Pose Transfer},
  author={Xue, Yu and Po, Lai-Man and Yu, Wing-Yin and Wu, Haoxuan and Xu, Xuyuan and Li, Kun and Liu, Yuyang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```


### Acknowledgments
Our code is based on [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention) and [PIDM](https://github.com/ankanbhunia/PIDM), thanks for their great works.