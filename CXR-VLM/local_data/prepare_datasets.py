# Partition code adapted from original GLORIA paper
# (https://github.com/marshuang80/gloria/blob/main/gloria/datasets/preprocess_datasets.py)

import pandas as pd
import numpy as np
import os
import sys
import tqdm
import argparse
import random
import time

sys.path.append(os.getcwd())
from sklearn.model_selection import train_test_split

DATASETS_PATH = "/media/juliosilva/HDETS/LIVIA/Datasets/Radiology/"

# RSNA Pneumonia
PNEUMONIA_DATA_DIR = DATASETS_PATH + "RSNA_PNEUMONIA/"
PNEUMONIA_ORIGINAL_TRAIN_CSV = PNEUMONIA_DATA_DIR + "stage_2_train_labels.csv"
PNEUMONIA_TRAIN_CSV = PNEUMONIA_DATA_DIR + "train.csv"
PNEUMONIA_VALID_CSV = PNEUMONIA_DATA_DIR + "val.csv"
PNEUMONIA_TEST_CSV = PNEUMONIA_DATA_DIR + "test.csv"
PNEUMONIA_IMG_DIR = PNEUMONIA_DATA_DIR + "stage_2_train_images"
PNEUMONIA_TRAIN_PCT = 0.7

# SIIM Pneumothorax
PNEUMOTHORAX_DATA_DIR = DATASETS_PATH + "SIIM_Pneumothorax/"
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR + "train-rle.csv"
PNEUMOTHORAX_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR + "train.csv"
PNEUMOTHORAX_VALID_CSV = PNEUMOTHORAX_DATA_DIR + "valid.csv"
PNEUMOTHORAX_TEST_CSV = PNEUMOTHORAX_DATA_DIR + "test.csv"
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR + "dicom-images-train"
PNEUMOTHORAX_IMG_SIZE = 1024
PNEUMOTHORAX_TRAIN_PCT = 0.7


def preprocess_pneumonia_data():
    test_fac = 0.30

    try:
        df = pd.read_csv(PNEUMONIA_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the RSNA Pneumonia dataset is \
            stored at {PNEUMONIA_DATA_DIR}"
        )

    # create bounding boxes
    def create_bbox(row):
        if row["Target"] == 0:
            return 0
        else:
            x1 = row["x"]
            y1 = row["y"]
            x2 = x1 + row["width"]
            y2 = y1 + row["height"]
            return [x1, y1, x2, y2]

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # create labels
    df["Normal"] = df["bbox"].apply(lambda x: 1 if x == None else 0)
    df["Pneumonia"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

    # no encoded pixels mean healthy
    df["Path"] = df["patientId"].apply(lambda x: "stage_2_train_images/" + (x + ".dcm"))

    df = df[["Path", "Normal", "Pneumonia", "bbox"]]

    idx_pneumonia = list(np.squeeze(np.argwhere(df["Pneumonia"].values == 1)))
    idx_normal = list(np.squeeze(np.argwhere(df["Pneumonia"].values == 0)))

    # Resample balanced dataset
    random.seed(0)
    idx_normal = random.sample(idx_normal, len(idx_pneumonia))
    df = df[df.index.isin(idx_normal + idx_pneumonia)]

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac, random_state=0)

    train_df.to_csv(PNEUMONIA_TRAIN_CSV)
    test_val_df.to_csv(PNEUMONIA_TEST_CSV)


def preprocess_covid_data():

    path_dataset = "COVID-19_Radiography_Dataset/"

    covid_path, n_covid_test, n_covid_train = "COVID/images/", 1500, 432
    normal_path, n_normal_test, n_normal_train = "Normal/images/", 1500, 1729

    files_covid = os.listdir(DATASETS_PATH + path_dataset + covid_path)
    files_normal = os.listdir(DATASETS_PATH + path_dataset + normal_path)

    files_covid = [covid_path + iFile for iFile in files_covid]
    files_normal = [normal_path + iFile for iFile in files_normal]

    # Test partition
    files_covid_test = files_covid[:n_covid_test]
    files_normal_test = files_normal[:n_normal_test]

    labels_covid = [1 for i in files_covid_test] + [0 for i in files_normal_test]
    labels_no_covid = [0 for i in files_covid_test] + [1 for i in files_normal_test]

    # Create table
    df = pd.DataFrame(list(zip(files_covid_test + files_normal_test, labels_covid, labels_no_covid)),
                      columns=['Path', 'COVID', 'Normal'])
    df.to_csv("covid_test.csv")

    # Train partition
    files_covid_train = files_covid[n_covid_test:n_covid_test+n_covid_train]
    files_normal_train = files_normal[n_normal_test:n_normal_test+n_normal_train]

    labels_covid = [1 for i in files_covid_train] + [0 for i in files_normal_train]
    labels_no_covid = [0 for i in files_covid_train] + [1 for i in files_normal_train]

    # Create table
    df = pd.DataFrame(list(zip(files_covid_train + files_normal_train, labels_covid, labels_no_covid)),
                      columns=['Path', 'COVID', 'Normal'])
    df.to_csv("covid_train.csv")


def preprocess_pneumothorax_data(test_fac=0.45):

    test_fac = 0.45

    try:
        df = pd.read_csv(PNEUMOTHORAX_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the SIIM Pneumothorax dataset is \
            stored at {PNEUMOTHORAX_DATA_DIR}"
        )

    # get image paths
    img_paths = {}
    for subdir, dirs, files in tqdm.tqdm(os.walk(PNEUMOTHORAX_IMG_DIR)):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                img_paths[file_id] = os.path.join(subdir.replace(DATASETS_PATH + "SIIM_Pneumothorax/", ""), f)

    # no encoded pixels mean healthy
    df["Pneumothorax"] = df.apply(
        lambda x: 0.0 if x[" EncodedPixels"] == " -1" else 1.0, axis=1
    )
    df["Path"] = df["ImageId"].apply(lambda x: img_paths[x])
    df = df[["Path", "Pneumothorax",  " EncodedPixels"]]

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Pneumothorax"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Pneumothorax"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Pneumothorax"].value_counts())

    train_df.to_csv(PNEUMOTHORAX_TRAIN_CSV)
    valid_df.to_csv(PNEUMOTHORAX_VALID_CSV)
    test_df.to_csv(PNEUMOTHORAX_TEST_CSV)


def partition_chexpert_5x200_data():

    CHEXPERT_COMPETITION_TASKS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    DATASETS_PATH = "/media/juliosilva/Seagate Portable Drive/LIVIA/Datasets/Radiology/"
    CHEXPERT_DATA_DIR = DATASETS_PATH + "CheXpert/"

    CHEXPERT_ORIGINAL_TRAIN_CSV = CHEXPERT_DATA_DIR + "train_cheXbert.csv"
    df_all = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    df_all = df_all.fillna(0)
    df_all = df_all[df_all["Frontal/Lateral"] == "Frontal"]

    CHEXPERT_ORIGINAL_TEST_CSV = CHEXPERT_DATA_DIR + "chexpert_5x200.csv"
    df_test = pd.read_csv(CHEXPERT_ORIGINAL_TEST_CSV)

    df_train = df_all[np.logical_not(np.isin(df_all["Path"], df_test["Path"]))]
    df_train.to_csv("chexpert_train.csv")


def adequate_chexpert_train_samples():
    import os
    import json
    import pandas as pd
    import numpy as np
    import glob
    import re
    import time

    DATASETS_PATH = "/media/juliosilva/Seagate Portable Drive/LIVIA/Datasets/Radiology/"
    PNEUMONIA_DATA_DIR = DATASETS_PATH + "CheXpert/"
    categories = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
                  'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                  'Fracture', 'Support Devices']

    table_train = pd.read_csv(PNEUMONIA_DATA_DIR + "chexpert_train.csv")

    data = []
    for i in range(len(table_train)):
        t1 = time.time()

        relative_path = table_train["Path"][i]

        if os.path.isfile(DATASETS_PATH + "CheXpert/CheXpert-v1.0/" + relative_path):

            data.append({"image": "CheXpert/CheXpert-v1.0/" + relative_path,
                         "prompts": [],
                         "prompts_categories": [],
                         "study_categories": [iCategory for iCategory in categories if table_train[iCategory][i]]})

        t2 = time.time()
        print(str(i) + "/" + str(len(table_train)) + " -- " + str(t2 - t1), end="\r")

    file = open('./vlp/datasets/partitions/CheXpert-train.txt', 'w')
    for iSample in data:
        file.write(json.dumps(iSample))
        file.write("\n")
    file.close()


def preprocess_mimic_5x200_data():

    CHEXPERT_COMPETITION_TASKS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    DATASETS_PATH = "/media/juliosilva/Seagate Portable Drive/LIVIA/Datasets/Radiology/"
    PNEUMONIA_DATA_DIR = DATASETS_PATH + "MIMIC-CXR-2/2.0.0/"

    table_partitions = pd.read_csv(PNEUMONIA_DATA_DIR + "mimic-cxr-2.0.0-split.csv")
    table_metadata = pd.read_csv(PNEUMONIA_DATA_DIR + "mimic-cxr-2.0.0-metadata.csv")
    table_labels_images = pd.read_csv(PNEUMONIA_DATA_DIR + "mimic-cxr-2.0.0-negbio.csv")

    # Select only train subset
    table_partitions = table_partitions[(table_partitions["split"] == "test")]

    # Select only chest PA radiographs
    table_metadata = table_metadata[(table_metadata["ViewPosition"] == "PA") | (table_metadata["ViewPosition"] == "AP")]
    table_partitions = table_partitions[np.in1d(list(table_partitions["dicom_id"]), list(table_metadata["dicom_id"]))]

    # Select the samples from partitions table to labels table
    table_partitions = table_partitions.reset_index()

    # % --------------------------------------------------------------------------
    # mimic_5x200

    data = []
    path, study_id_test = [], []
    Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural_Effusion = [], [], [], [], []
    images, studies, subjects = table_partitions["dicom_id"], table_partitions["study_id"], table_partitions["subject_id"]
    for i in range(len(table_partitions)):
        t1 = time.time()

        image_id = images[i]
        study_id = studies[i]
        subject_id = subjects[i]

        folder = str(subject_id)[:2]

        relative_path = "files/p" + folder + "/p" + str(subject_id) + "/s" + str(study_id) + "/" + image_id + ".jpg"
        if os.path.isfile(PNEUMONIA_DATA_DIR + relative_path):
            subTable_labels = table_labels_images[table_labels_images["study_id"] == study_id]
            subTable_labels = subTable_labels.fillna(0)

            if len(subTable_labels) > 0:

                path.append(relative_path)
                study_id_test.append(study_id)
                Atelectasis.append(subTable_labels["Atelectasis"].values.item())
                Cardiomegaly.append(subTable_labels["Cardiomegaly"].values.item())
                Consolidation.append(subTable_labels["Consolidation"].values.item())
                Edema.append(subTable_labels["Edema"].values.item())
                Pleural_Effusion.append(subTable_labels["Pleural Effusion"].values.item())

                t2 = time.time()
                print(str(i) + "/" + str(len(table_partitions)) + " -- " + str(t2-t1), end="\r")

    # Create table
    df = pd.DataFrame(list(zip(path, study_id_test, Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural_Effusion)),
                      columns=['Path', 'study_id', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'])

    task_dfs = []
    for i, t in enumerate(CHEXPERT_COMPETITION_TASKS):
        index = np.zeros(14)
        index[i] = 1
        df_task = df[
            (df["Atelectasis"] == index[0])
            & (df["Cardiomegaly"] == index[1])
            & (df["Consolidation"] == index[2])
            & (df["Edema"] == index[3])
            & (df["Pleural Effusion"] == index[4])
            ]
        df_task = df_task.sample(n=200)
        task_dfs.append(df_task)
    df_200 = pd.concat(task_dfs)

    df_200.to_csv("mimic_5x200.csv")

    # % --------------------------------------------------------------------------
    # mimic_pneumonia

    data = []
    path, study_id_test = [], []
    Pneumonia, NoFinding = [], []
    images, studies, subjects = table_partitions["dicom_id"], table_partitions["study_id"], table_partitions["subject_id"]
    for i in range(len(table_partitions)):
        t1 = time.time()

        image_id = images[i]
        study_id = studies[i]
        subject_id = subjects[i]

        folder = str(subject_id)[:2]

        relative_path = "files/p" + folder + "/p" + str(subject_id) + "/s" + str(study_id) + "/" + image_id + ".jpg"
        if os.path.isfile(PNEUMONIA_DATA_DIR + relative_path):
            subTable_labels = table_labels_images[table_labels_images["study_id"] == study_id]
            subTable_labels = subTable_labels.fillna(0)

            if len(subTable_labels) > 0:

                path.append(relative_path)
                study_id_test.append(study_id)
                Pneumonia.append(subTable_labels["Pneumonia"].values.item())
                NoFinding.append(subTable_labels["No Finding"].values.item())
                t2 = time.time()
                print(str(i) + "/" + str(len(table_partitions)) + " -- " + str(t2-t1), end="\r")

    # Create table
    df = pd.DataFrame(list(zip(path, study_id_test, NoFinding, Pneumonia)),
                      columns=['Path', 'study_id', 'No Finding', 'Pneumonia'])

    task_dfs = []
    for i, t in enumerate(['No Finding', 'Pneumonia']):
        index = np.zeros(14)
        index[i] = 1
        df_task = df[
            (df["Pneumonia"] == index[0])
            & (df["No Finding"] == index[1])
            ]
        df_task = df_task.sample(n=200)
        task_dfs.append(df_task)
    df_200 = pd.concat(task_dfs)

    df_200.to_csv("mimic_pneumonia.csv")

    return df_200