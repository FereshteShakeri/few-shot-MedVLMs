"""
Dataset and Dataloader preparation for vision-language pre-training
"""

import pandas as pd
import ast
import random

from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from cxrvlms.pretraining.data.dataset import Dataset, UniformDataset
from cxrvlms.pretraining.data.transforms import LoadImage, SelectRelevantKeys, ProduceDescription


def get_loader(dataframes_path, data_root_path, datasets, balance=False, batch_size=8, num_workers=0,
               banned_categories=None, caption="A radiology image of [CLS]",  memory_cache=0.0):

    """
    Dataloaders generation for vision-language pretraining. Read all dataframes from assembly model and combines
    them into a unified dataframe. Also, a dataloader is conditioned for training.
    """

    # Assembly dataframes into a combined data structure
    print("Setting assebly data...")
    data = []
    for iDataset in datasets:
        print("Processing data: " + iDataset)

        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv")

        for i in range(len(dataframe)):
            data_i = dataframe.loc[i, :].to_dict()

            # Remove banned words - for evaluating on incremental categories
            banned = False
            if banned_categories is not None:
                for iCat in data_i["study_categories"]:
                    for iiCat in banned_categories:
                        if iiCat in iCat:
                            banned = True
            if banned:
                continue

            # Add sample to general data
            data_i['study_categories'] = ast.literal_eval(data_i['study_categories'])
            data_i['prompts_categories'] = ast.literal_eval(data_i['prompts_categories'])
            data_i['prompts'] = ast.literal_eval(data_i['prompts'])
            data_i["image_name"] = data_i["image"]
            data_i["image_path"] = data_root_path + data_i["image"]
            data.append(data_i)

    data = data[0:300]
    print('Total assembly data samples: {}'.format(len(data)))

    # Test/Val partition
    i = list(range(len(data)))  # example list (please notice the explicit call to 'list')
    random.shuffle(i)  # shuffle the list
    idx_train, idx_val = i[int(len(data)/10):], i[:int(len(data)/10)]
    data_train, data_val = list(map(data.__getitem__, idx_train)), list(map(data.__getitem__, idx_val))
    print('Total training samples: {}'.format(len(data_train)))
    print('Total validation samples: {}'.format(len(data_val)))

    # Prepare data transforms for loader
    transforms = Compose([
        LoadImage(size=(224, 224), canvas=True, total_samples=len(data), memory_cache=memory_cache),
        ProduceDescription(caption=caption),
        SelectRelevantKeys()
    ])

    # Set data
    if balance:  # Balance accross datasets
        train_dataset = UniformDataset(data=data_train, transform=transforms)
        val_dataset = UniformDataset(data=data_val, transform=transforms)
    else:
        train_dataset = Dataset(data=data_train, transform=transforms)
        val_dataset = Dataset(data=data_val, transform=transforms)

    # Set dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True)

    # Set dataloaders in dict
    datalaoders = {"train": train_loader, "val": val_loader}

    return datalaoders
