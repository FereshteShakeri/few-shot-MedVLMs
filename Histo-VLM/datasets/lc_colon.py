import os
import copy
import math
import random
from collections import defaultdict, OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, listdir_nohidden



TO_BE_IGNORED = ["README.txt"]

class_names = ['Colon adenocarcinoma', 'Benign colonic tissue']

labels_dict = {"colon_aca": "Colon adenocarcinoma", "colon_n": "Benign colonic tissue"}

templates = ["a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image"]

class LCCOLON(DatasetBase):

    dataset_dir = "LC25000"

    def __init__(self, root, preprocess):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "colon")

        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        # classnames, labels = self.read_classnames(text_file)
        self.template = templates

        train, val, test = self.read_data()

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        data_count = 0
        for label, folder in enumerate(folders):

            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = labels_dict[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)   
                print(impath)
                print(label)
                print(classname)         
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
                data_count += 1
        print(data_count)
        
        split_ratio = int(data_count/8) 
        data_train = items[int(data_count/8):]
        data_test = items[:int(data_count/8)]
        random.shuffle(data_train)
        return data_train[:int(len(data_train)/2)], data_train[int(len(data_train)/2):], data_test

    def generate_fewshot_dataset_(self,num_shots, split):

            print('num_shots is ',num_shots)
            if split == "train":
                few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
            elif split == "val":
                few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
        
            return few_shot_data