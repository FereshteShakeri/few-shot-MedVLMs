import numpy as np
import random
import torch
import copy
import time

from PIL import Image
from torchvision.transforms import Resize
from kornia.augmentation import RandomHorizontalFlip, RandomAffine, ColorJitter


BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'

# Categories used for pretraining - important for label-dimension alignement
CATEGORIES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
              'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
              'Fracture', 'Support Devices']

# Augmentations for pretraining
augmentations_pretraining = torch.nn.Sequential(RandomHorizontalFlip(p=0.5),
                                                RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1)),
                                                ColorJitter(p=0.25, brightness=0.2, contrast=0.2))


# Function lo load an image from data dict, and scale to target size
def load_image(image_path, size, canvas):

    # Read image
    if "dcm" in image_path:
        import pydicom
        dicom = pydicom.read_file(image_path)
        img = np.array(dicom.pixel_array, dtype=float)
    else:
        img = Image.open(image_path)
        max_size = max(img.size)
        scale = max_size / size[0]
        img.draft('L', (img.size[0] / scale, img.size[1] // scale))
        img = np.asarray(img, dtype=float)

    # Scale intensity
    img /= 255.

    # Add channel
    img = np.expand_dims(img, 0)

    # Resize image
    img = torch.tensor(img)
    if not canvas or (img.shape[-1] == img.shape[-2]):
        img = Resize(size)(img)
    else:
        sizes = img.shape[-2:]
        max_size = max(sizes)
        scale = max_size / size[0]
        img = Resize((int(img.shape[-2] / scale), int((img.shape[-1] / scale)))).cuda()(img.cuda())
        img = torch.nn.functional.pad(img,
                                      (0, size[0] - img.shape[-1], 0, size[1] - img.shape[-2], 0, 0))
    img = img.cpu().numpy()
    return img


class LoadImage():

    def __init__(self, size=(224, 224), canvas=True, total_samples=300000, memory_cache=0.0):
        self.size = size
        self.canvas = canvas
        self.counter = 0
        self.total_samples = total_samples
        self.memory_cache = memory_cache

    def __call__(self, data):

        # If we are using cache memory
        if self.memory_cache > 0.0:
            # Check if image has already been loaded and pre-processed
            if "cache" in data.keys():
                d = copy.deepcopy(data)
                d["image"] = np.float32(d["image"]) / 255.
            # Otherwise, load image
            else:
                img = load_image(data['image_path'], self.size, self.canvas)
                # Check for space in cache memory
                if self.counter < (self.total_samples*self.memory_cache):
                    self.counter += 1
                    data["image"], data["cache"] = np.uint8((img * 255)), True
                d = copy.deepcopy(data)
                d["image"] = img
        else:
            img = load_image(data['image_path'], self.size, self.canvas)
            d = copy.deepcopy(data)
            d["image"] = img

        # Add channels to grayscale image
        if d["image"].shape[0] == 1:
            d["image"] = np.repeat(d["image"], 3, 0)
        return d


class ProduceDescription():
    def __init__(self, caption):
        self.caption = caption

    def __call__(self, data):
        if "MIMIC" in data["image_path"]:
            idx = random.sample(list(np.arange(0, len(data["prompts"]))), 1)[0]

            data["prompt_selected"] = [data["prompts"][idx]]
            category_prompt_selected = np.array([iCategory in data['prompts_categories'][idx] for iCategory in CATEGORIES], dtype=int)
            data["category_prompt_selected"] = category_prompt_selected

        elif "CheXpert" in data["image_path"]:
            if len(data["study_categories"]) == 0:
                data["study_categories"] = ["Other finding"]

            idx = random.sample(list(np.arange(0, len(data["study_categories"]))), 1)[0]

            data["prompt_selected"] = [self.caption.replace("[CLS]",  data["study_categories"][idx])]
            category_prompt_selected = np.array([iCategory == data["study_categories"][idx] for iCategory in CATEGORIES], dtype=int)
            data["category_prompt_selected"] = category_prompt_selected

        data["study_categories"] = np.array([iCategory in data["study_categories"] for iCategory in CATEGORIES], dtype=int)

        return data


class SelectRelevantKeys():
    def __call__(self, data):
        d = {key: data[key] for key in ['image', 'study_categories', 'prompt_selected', 'category_prompt_selected']}
        return d