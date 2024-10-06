

## Large Vision-Language Models (VLMs)  Adaptation for Histology (Histo)

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash

conda create -n histo python=3.7
conda activate histo

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```


## Get Started
### Configs
Specify basic configuration as (num_shots, num_tasks, method, etc) and hyperparameters in `configs/base.yaml`. 

### Experiments


For each datasets:

```bash
python main.py --base_config configs/base.yaml --dataset_config configs/{dataset_name}.yaml
```

Example of running LP++ on nct dataset in 16 shot setting:
```bash
python main.py --base_config configs/base.yaml --dataset_config configs/nct.yaml --opt root_path {DATA_PATH} output_dir {OUTPUT_PATH} method LinearProbe_P2  shots 16 tasks 5
```


[This repo is built on top of [LP++](https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline).]

