import os
import copy
import math
import random
import pandas as pd
from collections import defaultdict, OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, listdir_nohidden


labels = ['nontumor_skin_necrosis_necrosis', 'tumor_skin_naevus_naevus', 'nontumor_skin_muscle_skeletal',
          'tumor_skin_epithelial_sqcc', 'nontumor_skin_sweatglands_sweatglands', 'nontumor_skin_vessel_vessel',
          'nontumor_skin_elastosis_elastosis', 'nontumor_skin_chondraltissue_chondraltissue', 'nontumor_skin_hairfollicle_hairfollicle',
          'nontumor_skin_epidermis_epidermis', 'nontumor_skin_nerves_nerves', 'nontumor_skin_subcutis_subcutis',
          'tumor_skin_melanoma_melanoma', 'tumor_skin_epithelial_bcc', 'nontumor_skin_dermis_dermis',
          'nontumor_skin_sebaceousglands_sebaceousglands']

TO_BE_IGNORED = ["README.txt"]

class_names = ['Necrosis', 'Skeletal muscle', 'Eccrine sweat glands',
               'Vessels', 'Elastosis', 'Chondral tissue', 'Hair follicle',
               'Epidermis', 'Nerves', 'Subcutis', 'Dermis', 'Sebaceous glands',
               'Squamous-cell carcinoma', 'Melanoma in-situ',
               'Basal-cell carcinoma', 'Naevus']

labels_dict = {"ADI": "Adipose", "BACK": "Background", "DEB": "Debris", 
               "LYM": "Lymphocytes", "MUC": "Mucus", "MUS": "Smooth muscle",
               "NORM": "Normal colon mucosa", "STR": "Cancer-associated stroma", 
                "TUM": "Colorectal adenocarcinoma epithelium"}

templates = ["a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image"]


class SKIN(DatasetBase):

    dataset_dir = "skincancer/data"

    def __init__(self, root, preprocess):
        
        self.root_dir = os.path.join(root, "skincancer")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "tiles")
        
        split_file = os.path.join(self.dataset_dir, 'tiles-v2.csv')
        
        self.data = pd.read_csv(split_file)
        
        data_train = self.data[self.data['set'] == 'Train']
        data_val = self.data[self.data['set'] == "Validation"]
        data_test = self.data[self.data['set'] == 'Test']
        
        self.image_paths = self.data['file'].values
        self.labels = self.data['class'].values
        
        self.cat_to_num_map = {'nontumor_skin_necrosis_necrosis': 0,
                               'nontumor_skin_muscle_skeletal': 1,
                               'nontumor_skin_sweatglands_sweatglands': 2,
                               'nontumor_skin_vessel_vessel': 3,
                               'nontumor_skin_elastosis_elastosis': 4,
                               'nontumor_skin_chondraltissue_chondraltissue': 5,
                               'nontumor_skin_hairfollicle_hairfollicle': 6,
                               'nontumor_skin_epidermis_epidermis': 7,
                               'nontumor_skin_nerves_nerves': 8,
                               'nontumor_skin_subcutis_subcutis': 9,
                               'nontumor_skin_dermis_dermis': 10,
                               'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                               'tumor_skin_epithelial_sqcc': 12,
                               'tumor_skin_melanoma_melanoma': 13,
                               'tumor_skin_epithelial_bcc': 14,
                               'tumor_skin_naevus_naevus': 15
                               }

        self.tumor_map = {'tumor_skin_epithelial_sqcc': 0,
                          'tumor_skin_melanoma_melanoma': 1,
                          'tumor_skin_epithelial_bcc': 2,
                          'tumor_skin_naevus_naevus': 3
                          }
        # if tumor:
        #     self.data = self.data[self.data['malignicy'] == 'tumor']
        # self.tumor = tumor

        self.template = templates

        train = self.read_data(data_train)
        val = self.read_data(data_val)
        test = self.read_data(data_test)


        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, data_split):

        # folders = listdir_nohidden(image_dir, sort=True)
        # folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []
        for index, row in data_split.iterrows():
            impath = os.path.join(self.root_dir, row['file'])
            item = Datum(impath=impath, label=row['class_int'], classname=row['class'])
            items.append(item)
            
        return items


    def generate_fewshot_dataset_(self,num_shots, split):

            print('num_shots is ',num_shots)
            if split == "train":
                few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
            elif split == "val":
                few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
        
            return few_shot_data