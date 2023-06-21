# HaLP: Hallucinating Latent Positives for Skeleton-based Self-Supervised Learning of Actions 
![](teaser.jpg)

## Setup environment
- `conda create -n halp python=3`
- `conda activate halp`
- `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`
- `pip install tqdm geomstats termcolor wandb`

## Data preprocessing
- Follow steps from [Skeleton Contrast](https://github.com/fmthoker/skeleton-contrast)

## Update paths, setup wandb
- Update data paths (`dataroot`) in `options/options_*` 
- Set project name (`wandb.init`) in `pretrain_moco_single_modality.py` and `action_classification.py`

## Training scripts
- We pretrain the model with standard MoCo (single modality) for 200 (NTU) and (500) epochs respectively. 
- Download the 200 and 500 epoch pretrained baseline models from [here](https://www.cis.jhu.edu/~ashah/HaLP/baseline_checkpoints/)
- Refer to `scripts` for training and evaluation.

## Citatiion
If you find this repository useful in your research, please cite:
```
@InProceedings{Shah_2023_CVPR,
    author    = {Shah, Anshul and Roy, Aniket and Shah, Ketul and Mishra, Shlok and Jacobs, David and Cherian, Anoop and Chellappa, Rama},
    title     = {HaLP: Hallucinating Latent Positives for Skeleton-Based Self-Supervised Learning of Actions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {18846-18856}
}
```

## Acknowledgements
- This code is based on [CMD](https://github.com/maoyunyao/CMD)'s official code. 