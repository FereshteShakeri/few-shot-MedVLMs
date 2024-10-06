"""
This script contains relative or absolute paths for local data
(i.e. dataframes for pretraining/transferability, data paths,
 and results paths)
"""

# Path with datasets
PATH_DATASETS = "/mnt/data1/Radiology/data/"
# PATH_DATASETS = "/projets/Jsilva/datasets/Radiology/"

# Path with pretraining and transferability dataframes
PATH_DATAFRAME_PRETRAIN = "./local_data/dataframes/pretraining/"
PATH_DATAFRAME_TRANSFERABILITY = "./local_data/dataframes/transferability/"
PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION = PATH_DATAFRAME_TRANSFERABILITY + "classification/"
PATH_DATAFRAME_TRANSFERABILITY_SEGMENTATION = PATH_DATAFRAME_TRANSFERABILITY + "segmentation/"

# Paths for results
PATH_RESULTS_PRETRAIN = "./local_data/results/pretraining/"
PATH_RESULTS_TRANSFERABILITY = "./local_data/results/transferability/"
