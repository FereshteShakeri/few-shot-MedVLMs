"""
Script to retrieve transferability experiments setting
(i.e. dataframe path, target classes, and task type)
"""

from cxrvlms.modeling.prompts import *


def get_experiment_setting(experiment):

    # Transferability for classification
    if experiment == "chexpert_5x200":
        setting = {"experiment": "chexpert_5x200",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "./local_data/dataframes/transferability/classification/chexpert_5x200.csv",
                   "base_samples_path": "CheXpert/CheXpert-v1.0/",
                   "prompt_generator": generate_chexpert_class_prompts,
                   "task": "classification",
        }
    elif experiment == "mimic_5x200":
        setting = {"experiment": "mimic_5x200",
                   "targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "./local_data/dataframes/transferability/classification/mimic_5x200.csv",
                   "base_samples_path": "MIMIC-CXR-2/2.0.0/",
                   "prompt_generator": generate_chexpert_class_prompts,
                   "task": "classification",
        }
    elif experiment == "mimic_pneumonia":
        setting = {"experiment": "mimic_pneumonia",
                   "targets": ["No Finding", "Pneumonia"],
                   "dataframe": "./local_data/dataframes/transferability/classification/mimic_pneumonia.csv",
                   "base_samples_path": "MIMIC-CXR-2/2.0.0/",
                   "prompt_generator": generate_rsna_class_prompts,
                   "task": "classification",
                   }
    elif experiment == "covid_train_2class":
        setting = {"experiment": "covid_train_2class",
                "targets": ["Normal", "COVID"],
                "dataframe": "./local_data/dataframes/transferability/classification/covid_train.csv",
                "base_samples_path": "COVID-19_Radiography_Dataset/",
                "prompt_generator": generate_covid_class_prompts,
                "task": "classification",
                }
    elif experiment == "covid_test_2class":
        setting = {"experiment": "covid_test_2class",
                "targets": ["Normal", "COVID"],
                "dataframe": "./local_data/dataframes/transferability/classification/covid_test.csv",
                "base_samples_path": "COVID-19_Radiography_Dataset/",
                "prompt_generator": generate_covid_class_prompts,
                "task": "classification",
                }
    elif experiment == "covid_train_4class":
        setting = {"experiment": "covid_train_4class_new",
                "targets": ["Normal", "COVID", "Pneumonia", "Lung Opacity"],
                "dataframe": "./local_data/dataframes/transferability/classification/covid_train.csv",
                "base_samples_path": "COVID-19_Radiography_Dataset/",
                "prompt_generator": generate_covid_class_prompts,
                "task": "classification",
                }
    elif experiment == "covid_test_4class":
        setting = {"experiment": "covid_test_4class",
                "targets": ["Normal", "COVID", "Pneumonia", "Lung Opacity"],
                "dataframe": "./local_data/dataframes/transferability/classification/covid_test.csv",
                "base_samples_path": "COVID-19_Radiography_Dataset/",
                "prompt_generator": generate_covid_class_prompts,
                "task": "classification",
                }
    elif experiment == "rsna_pneumonia_train":
        setting = {"experiment": "RSNA_pneumonia_train",
                   "targets": ["Normal", "Pneumonia"],
                   "dataframe": "./local_data/dataframes/transferability/classification/rsna_pneumonia_train.csv",
                   "base_samples_path": "RSNA_PNEUMONIA/",
                   "prompt_generator": generate_rsna_class_prompts,
                   "task": "classification",
        }
    elif experiment == "rsna_pneumonia_test":
        setting = {"experiment": "RSNA_pneumonia_test",
                   "targets": ["Normal", "Pneumonia"],
                   "dataframe": "./local_data/dataframes/transferability/classification/rsna_pneumonia_test.csv",
                   "base_samples_path": "RSNA_PNEUMONIA/",
                   "prompt_generator": generate_rsna_class_prompts,
                   "task": "classification",
                   }
    else:
        setting = None
        print("Experiment not prepared...")

    return setting