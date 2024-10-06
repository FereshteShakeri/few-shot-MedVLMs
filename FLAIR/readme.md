## Large Vision-Language Models (VLMs)  Adaptation for Ophthalmology ()


[This repo is built on top of [FLAIR](https://github.com/jusiro/FLAIR).]
Please check the instructions of FLAIR below here.


## Install FLAIR

* Install in your enviroment a compatible torch version with your GPU. For example:
```
conda create -n flair_env python=3.8 -y
conda activate flair_env
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

* Install FLAIR library.
```
pip install git+https://github.com/jusiro/FLAIR.git
```

## Usage

```
from PIL import Image
import numpy as np

# Import FLAIR
from flair import FLAIRModel

# Set model
model = FLAIRModel(from_checkpoint=True)

# Load image and set target categories 
# (if the repo is not cloned, download the image and change the path!)

image = np.array(Image.open("./documents/sample_macular_hole.png"))
text = ["normal", "healthy", "macular edema", "diabetic retinopathy", "glaucoma", "macular hole",
        "lesion", "lesion in the macula"]

# Forward FLAIR model to compute similarities
probs, logits = model(image, text)

print("Image-Text similarities:")
print(logits.round(3)) # [[-0.32  -2.782  3.164  4.388  5.919  6.639  6.579 10.478]]
print("Probabilities:")
print(probs.round(3))  # [[0.      0.     0.001  0.002  0.01   0.02   0.019  0.948]]
```

## Pre-training and transferability

In the following, we present the scripts for model pre-training and transferability. To use them, we recommend cloning the whole repository.

```
git clone https://github.com/jusiro/FLAIR.git
cd FLAIR
pip install -r requirements.txt
```


### 📦 Transferability to downstream tasks/domains
* Define the relative paths for datasets and dataframes in `./local_data/constants.py`.
* Prepare the experiment setting for the target dataset - we used `./local_data/experiments.py` to store them.

```
if experiment == "02_MESSIDOR":
    setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "02_MESSIDOR.csv",
               "task": "classification",
               "targets": {"no diabetic retinopathy": 0,
                           "mild diabetic retinopathy": 1,
                           "moderate diabetic retinopathy": 2,
                           "severe diabetic retinopathy": 3,
                           "proliferative diabetic retinopathy": 4}}
```

* Zero-shot (no adaptation).

```
python main_transferability.py --experiment 02_MESSIDOR --method zero_shot --load_weights True --domain_knowledge True  --shots_train 0% --shots_test 100% --project_features True --norm_features True --folds 1 
```

* Linear Probing.

```
python main_transferability.py --experiment 02_MESSIDOR --method lp --load_weights True --shots_train 16 --shots_val 16--shots_test 20% --project_features False --norm_features False --folds 5 
```

* Linear Probe + text

```
python main_transferability.py --experiment 02_MESSIDOR --method linearprobe_p2 --load_weights True --shots_train 16 --shots_val 16  --shots_test 20% --project_features False --norm_features False --folds 5 
```

# Citation

If you find this repository useful, please consider citing this paper:
```
@article{FLAIR2023,
  title={A Foundation LAnguage-Image model of the Retina (FLAIR): Encoding expert knowledge in text supervision},
  author={Julio Silva-Rodriguez and Hadi Chakor and Riadh Kobbi and Jose Dolz and Ismail Ben Ayed},
  journal={ArXiv Preprint},
  year={2023}
}
```

# License

- **Code and Model Weights** are released under [Apache 2.0 license](LICENSE)
