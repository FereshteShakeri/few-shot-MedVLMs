import os
import copy
import math
import random
from collections import defaultdict, OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, listdir_nohidden


labels = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

TO_BE_IGNORED = ["README.txt"]

class_names = ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle",
           "Normal colon mucosa", "Cancer-associated stroma", 
           "Colorectal adenocarcinoma epithelium"]

labels_dict = {"ADI": "Adipose", "BACK": "Background", "DEB": "Debris", 
               "LYM": "Lymphocytes", "MUC": "Mucus", "MUS": "Smooth muscle",
               "NORM": "Normal colon mucosa", "STR": "Cancer-associated stroma", 
                "TUM": "Colorectal adenocarcinoma epithelium"}

templates = ["a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image"]


class NCT(DatasetBase):

    dataset_dir = "NCT-CRC-HE-100K"

    def __init__(self, root, preprocess):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "NCT-CRC-HE-100K")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames, labels = self.read_classnames(text_file)
        self.template = templates
        train_path = os.path.join(self.dataset_dir, "NCT-CRC-HE-100K")
        test_path = os.path.join(self.dataset_dir, "CRC-VAL-HE-7K")

        train, val = self.read_data(train_path, "train")
        test,_ = self.read_data(test_path, "test")

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, data_path, split):
        image_dir = data_path
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        data_count = 0
        for label, folder in enumerate(folders):

            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = labels_dict[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)            
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                data_count += 1
        print(data_count)        
        if split == "train":
            random.shuffle(items)
            print(len(items[:int(data_count/2)]))
            print(len(items[int(data_count/2):]))
            return items[:int(data_count/2)], items[int(data_count/2):]
        elif split == "test":
            return items, items


    def generate_fewshot_dataset_(self,num_shots, split):

            print('num_shots is ',num_shots)
            if split == "train":
                few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
            elif split == "val":
                few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
        
            return few_shot_data