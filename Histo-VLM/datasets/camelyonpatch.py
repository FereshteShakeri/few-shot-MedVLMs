import os
import copy
import math
import random
import h5py
from collections import defaultdict, OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, listdir_nohidden


class_names = ['Lymph node', 'Lymph node containing metastatic tumor tissue']

templates = ["a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image"]


class PCAM(DatasetBase):

    dataset_dir = "camelyonpatch"

    def __init__(self, root, preprocess):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # self.image_dir = os.path.join(self.dataset_dir, "NCT-CRC-HE-100K")

        self.template = templates
        train_images = os.path.join(self.dataset_dir, "camelyonpatch_level_2_split_train_x.h5")
        train_labels = os.path.join(self.dataset_dir, "camelyonpatch_level_2_split_train_y.h5")
        val_images = os.path.join(self.dataset_dir, "camelyonpatch_level_2_split_valid_x.h5")
        val_labels = os.path.join(self.dataset_dir, "camelyonpatch_level_2_split_valid_y.h5")
        test_images = os.path.join(self.dataset_dir, "camelyonpatch_level_2_split_test_x.h5")
        test_labels = os.path.join(self.dataset_dir, "camelyonpatch_level_2_split_test_y.h5")

        train = self.read_data(train_images, train_labels)
        val = self.read_data(val_images, val_labels)
        test = self.read_data(test_images, test_labels)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, images_file, labels_file):

        items = []

        data_count = 0
        with h5py.File(images_file, 'r') as hf:
            #for k,v in hf.items():
                # if isinstance(v, h5py.Dataset):
                #     print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            print(hf['x'])
            # for image in hf['x']:
                # print(image.shape)
        """
        with h5py.File(labels_file, 'r') as hf:
            #for k,v in hf.items():
                # if isinstance(v, h5py.Dataset):
                #     print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            print(hf['y'])
            for label in hf['y']:
                print(label)
        """
        print(2/0)
        """
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
        """

    def generate_fewshot_dataset_(self,num_shots, split):

            print('num_shots is ',num_shots)
            if split == "train":
                few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
            elif split == "val":
                few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
        
            return few_shot_data