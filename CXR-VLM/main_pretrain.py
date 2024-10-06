
"""
Main function to pretrain VLMs model using
an assembly dataset and vision-text modalities.
"""

import argparse

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

from cxrvlms.pretraining.data.dataloader import get_loader
from cxrvlms.pretraining.data.transforms import augmentations_pretraining
from cxrvlms.modeling.model import VLMModel

from local_data.constants import *


def process(args):

    # Set data for training
    datalaoders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                             datasets=args.datasets, balance=args.balance, batch_size=args.batch_size,
                             num_workers=args.num_workers, banned_categories=args.banned_categories,
                             caption=args.caption, memory_cache=args.memory_cache)

    # Init FLAIR model
    model = VLMModel(vision_type=args.architecture, out_path=args.out_path, from_checkpoint=False,
                     vision_pretrained=True)

    # Training
    model.fit(datalaoders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num, transforms=augmentations_pretraining)


def main():
    parser = argparse.ArgumentParser()

    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN)
    parser.add_argument('--datasets', default=["CheXpert-train", "MIMIC-CXR-2-train"])
    parser.add_argument('--banned_categories', default=[])
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN, help='output path')

    # Prompts setting and augmentation hyperparams
    parser.add_argument('--caption', default="A radiology image of [CLS]")

    # Dataloader setting
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--memory_cache', default=0.0, type=float, help='memory_cache')

    # Training options
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--store_num', default=5, type=int)

    # Architecture and pretrained weights options
    parser.add_argument('--architecture', default='resnet_v2', help='resnet_v1 -- efficientnet')

    # Resources
    parser.add_argument('--num_workers', default=0, type=int, help='workers number for DataLoader')

    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()