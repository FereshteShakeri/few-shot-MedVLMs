## Large Vision-Language Models (VLMs) Adaptation for Chest X-rays (CXR)

[This repo is built on top of [CXR-VLM](https://github.com/jusiro/CXR-VLMs).]
Please check the instructions below here.
## Install

* Install in your enviroment a compatible torch version with your GPU. For example:
```
conda create -n vlms_env python=3.8 -y
conda activate vlms_env
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

```
git clone https://github.com/jusiro/CXR-VLMs.git
cd CXR-VLMs
pip install -r requirements.txt
```

## **Note**: problems during automatic **pre-trained weights download**

If you encounter any issue while downloading the **pre-trained weights** (i.e. `from_checkpoint=True`), you can manually download the weights from the following links (see Table), unzip the file, and store them at: `./flair/modeling/pretrained_weights/[ID].pth`.

| Backbone  |     ID      |                                                                                            |
|-----------|:-----------:|:------------------------------------------------------------------------------------------:|
| ResNet-50 | cxr_resnet  | [LINK](https://drive.google.com/file/d/1Lzgj8LORf-4stmQeUT6xlex0LHZ1U3rc/view?usp=sharing) |

### 📦 Transferability to downstream tasks/domains
* Define the relative paths for datasets and dataframes in `./local_data/constants.py`.
* Prepare the experiment setting for the target dataset - we used `./local_data/experiments.py` to store them.

```
if experiment == "chexpert_5x200":
        setting = {"experiment": "chexpert_5x200",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "./local_data/dataframes/transferability/classification/chexpert_5x200.csv",
                   "base_samples_path": "CheXpert/CheXpert-v1.0/",
                   "prompt_generator": generate_chexpert_class_prompts,
                   "task": "classification",
```

* Zero-shot (no adaptation).

```
python main_transferability.py --experiment chexpert_5x200 --method zero_shot --load_weights True --ensemble True --shots_train 80% --shots_test 20% --folds 5 
python main_transferability.py --experiment mimic_5x200 --method zero_shot --load_weights True --ensemble True --shots_train 80% --shots_test 20%  --folds 5 
python main_transferability.py --experiment covid_train --method zero_shot --load_weights True --ensemble True --shots_train 100% --shots_test 0% --experiment_test covid_test --folds 1 
python main_transferability.py --experiment rsna_pneumonia_train --method zero_shot --load_weights True --ensemble True --shots_train 100% --shots_test 0% --experiment_test rsna_pneumonia_test --folds 1 
```



* Linear Probing (Radford et al. (2021)) - few shot adaptation. Example of 16-shot results.

```
python main_transferability.py --experiment chexpert_5x200 --method lp --load_weights True --ensemble True --shots_train 16 --shots_test 20% --folds 5 
python main_transferability.py --experiment mimic_5x200 --method lp --load_weights True --ensemble True --shots_train 16 --shots_test 20%  --folds 5 
python main_transferability.py --experiment covid_train --method lp --load_weights True --ensemble True --shots_train 16 --shots_test 0% --experiment_test covid_test --folds 5
python main_transferability.py --experiment rsna_pneumonia_train --method lp --load_weights True --ensemble True --shots_train 16 --shots_test 0% --experiment_test rsna_pneumonia_test --folds 5  
```

* Linear Probing + text  - few shot adaptation. Example of 16-shot results.

```
python main_transferability.py --experiment chexpert_5x200 --method linearprobe_p2 --load_weights True --ensemble True --shots_train 16 --shots_test 20% --folds 5 
python main_transferability.py --experiment mimic_5x200 --method linearprobe_p2 --load_weights True --ensemble True --shots_train 16 --shots_test 20%  --folds 5 
python main_transferability.py --experiment covid_train --method linearprobe_p2 --load_weights True --ensemble True --shots_train 16 --shots_test 0% --experiment_test covid_test --folds 5
python main_transferability.py --experiment rsna_pneumonia_train --method linearprobe_p2 --load_weights True --ensemble True --shots_train 16 --shots_test 0% --experiment_test rsna_pneumonia_test --folds 5  
```