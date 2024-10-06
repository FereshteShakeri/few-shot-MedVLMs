import os
import random
import pandas as pd

from .utils import Datum, DatasetBase

templates = ["a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image"]

class SicapV2(DatasetBase):
    
    image_dir = "SICAPv2"
    
    def __init__(self, root, num_shots):

        self.image_dir = os.path.join(root, self.image_dir)

        csv_file = os.path.join(self.image_dir, "partition/Test", "Train.xlsx")
        self.data_train = pd.read_excel(csv_file)
        
        csv_file = os.path.join(self.image_dir, "partition/Test", "Test.xlsx")
        self.data_test = pd.read_excel(csv_file)

        # drop all columns except image_name and the label columns
        label_columns = ['NC', 'G3', 'G4', 'G5']  # , 'G4C']
        self.data_train = self.data_train[['image_name'] + label_columns]
        self.data_test = self.data_test[['image_name'] + label_columns]

        # get the index of the maximum label value for each row
        self.data_train['labels'] = self.data_train[label_columns].idxmax(axis=1)
        self.data_test['labels'] = self.data_test[label_columns].idxmax(axis=1)

        # replace the label column values with categorical values
        self.cat_to_num_map = label_map = {'NC': 0, 'G3': 1, 'G4': 2, 'G5': 3}  # , 'G4C': 4}
        self.data_train['labels'] = self.data_train['labels'].map(label_map)
        self.data_test['labels'] = self.data_test['labels'].map(label_map)

        self.image_paths_train = self.data_train['image_name'].values
        self.labels_train = self.data_train['labels'].values
        self.image_paths_test = self.data_test['image_name'].values
        self.labels_test = self.data_test['labels'].values
        self.classes = ["non-cancerous well-differentiated glands",
                "gleason grade 3 with atrophic well differentiated and dense glandular regions",
                "gleason grade 4 with cribriform, ill-formed, large-fused and papillary glandular patterns",
                "gleason grade 5 with nests of cells without lumen formation, isolated cells and pseudo-roseting patterns",
                ]

        self.template =templates
        # self.image_dir = image_dir
        # self.transform = transform
        # self.train = train
        train, val = self.split_data(self.data_train, "train")
        test,_ = self.split_data(self.data_test, "test")

        
        super().__init__(train_x=train, val=val, test=test)
        
    def split_data(self, data_split, split):
        items = []
        print(self.data_train)
        for i in range(len(data_split)):
            impath = os.path.join(self.image_dir, 'images', data_split.at[i, "image_name"])
            item = Datum(
                impath=impath,
                label=int(data_split.at[i, "labels"]),
                classname=self.classes[data_split.at[i, "labels"]]
            )
            items.append(item)
        
        print(items)
        if split == "train":
            random.shuffle(items)
            print(len(items[:int(len(data_split)/2)]))
            print(len(items[int(len(data_split)/2):]))
            return items[:int(len(data_split)/2)], items[int(len(data_split)/2):]
        elif split == "test":
            return items, items

    """
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        return image, label
    
    def read_data(self, classnames, labels):
        for i in range(len(classnames)):
            item = Datum(impath=impath, label=labels[folder], classname=classname)
            items.append(item)
            
        return items
    """
    def generate_fewshot_dataset_(self,num_shots, split):

        print('num_shots is ',num_shots)
        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)

        return few_shot_data