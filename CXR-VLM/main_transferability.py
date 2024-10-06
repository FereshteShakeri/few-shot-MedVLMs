
"""
Main function to transfer pretrained FLAIR model
to downstream classification and segmentation tasks.
It includes fine-tuning, linear probing, and vision-language
adapters. Also, it allows to directly testing zero-shot
generalization trough text prompts.
"""

import argparse
import torch

from cxrvlms.modeling.model import VLMModel
from cxrvlms.transferability.data.dataloader import get_dataloader_splits
from cxrvlms.utils.metrics import evaluate, average_folds_results, save_results
from cxrvlms.modeling.misc import set_seeds
from cxrvlms.modeling.prompts import generate_covid_class_prompts, generate_rsna_class_prompts, generate_class_prompts, generate_chexpert_class_prompts
from cxrvlms.transferability.modeling.adapters import LinearProbe, ClipAdapter, ZeroShot, TipAdapter, LinearProbe2, CoOp, KgCoOp, CoCoOp
from cxrvlms.transferability.modeling.finetuning import FineTuning

from local_data.constants import *
from local_data.experiments import get_experiment_setting

import warnings
warnings.filterwarnings("ignore")

set_seeds(42, use_cuda=torch.cuda.is_available())


def init_adapter(model, args):

    if "FT" in args.method:
        print("Transferability by Fine-tuning...", end="\n")
        adapter = FineTuning(model, args.setting["targets"], args.method, tta=args.tta, fta=args.fta,
                             loaders=args.loaders, epochs=args.epochs, update_bn=args.update_bn,
                             freeze_classifier=args.freeze_classifier, last_lp=args.last_lp, lr=args.lr,
                             task=args.setting["task"], save_best=args.save_best, patience=args.patience)
    elif args.method == "lp":
        print("Transferability by Linear Probing...", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)
    elif args.method == "clipAdapter":
        print("Transferability by CLIP Adapter...", end="\n")
        adapter = ClipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                              ensemble=args.ensemble, prompt_generator=prompt_generator(args))
    elif args.method == "tipAdapter":
        print("Transferability by TIP-Adapter Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                             ensemble=args.ensemble, prompt_generator=prompt_generator(args), train=False)
    elif args.method == "tipAdapter-f":
        print("Transferability by TIP-Adapter-f Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                             ensemble=args.ensemble, prompt_generator=prompt_generator(args), train=True)
    elif args.method == "coop":
        print("Transferability by CLIP Adapter...", end="\n")
        adapter = CoOp(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                              ensemble=args.ensemble, prompt_generator=prompt_generator(args))
    elif args.method == "kgcoop":
        print("Transferability by CLIP Adapter...", end="\n")
        adapter = KgCoOp(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                              ensemble=args.ensemble, prompt_generator=prompt_generator(args))
    elif args.method == "cocoop":
        print("Transferability by CLIP Adapter...", end="\n")
        adapter = CoCoOp(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                              ensemble=args.ensemble, prompt_generator=prompt_generator(args))
    elif args.method == "linearprobe_p2":
        print("Transferability by TIP-Adapter-f Adapter...", end="\n")
        adapter = LinearProbe2(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                             ensemble=args.ensemble, prompt_generator=prompt_generator(args), train=True)
    elif args.method == "zero_shot":
        print("Zero-shot classification...", end="\n")
        adapter = ZeroShot(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                           ensemble=args.ensemble, prompt_generator=args.setting["prompt_generator"])
    else:
        print("Adapter not implemented... using LP", end="\n")
        adapter = LinearProbe(args, model.vision_model)

    return adapter


def generate_experiment_id(args):
    id = args.experiment + '_vision_' + args.architecture + '_method_' + args.method + '_pretrained_' \
         + str(args.load_weights) + '_shots_train_' + args.shots_train + '_shots_test_' + args.shots_test + \
         '_balance_' + str(args.balance)
    return id

def prompt_generator(args):
    if args.experiment == "rsna_pneumonia_train":
        return generate_rsna_class_prompts
    elif args.experiment == "covid_train":
        return generate_covid_class_prompts
    else:
        return generate_chexpert_class_prompts

def process(args):

    # KFold cross-validation
    args.metrics_test, args.metrics_external, args.weights = [], [[] for i in range(len(args.experiment_test))], []
    for iFold in range(args.folds):
        print("\nTransferability (fold : " + str(iFold + 1) + ")", end="\n")
        args.iFold = iFold

        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
        args.setting = get_experiment_setting(args.experiment)

        # Init FLAIR model
        model = VLMModel(vision_type=args.architecture, from_checkpoint=args.load_weights,
                         weights_path=args.weights_path, projection=args.project_features,
                         norm_features=args.norm_features, vision_pretrained=args.init_imagenet)

        # Set datasets
        args.loaders = get_dataloader_splits(args.setting["dataframe"],
                                             args.data_root_path + args.setting["base_samples_path"],
                                             args.setting["targets"],
                                             shots_train=args.shots_train, shots_val=args.shots_val,
                                             shots_test=args.shots_test, balance=args.balance,
                                             batch_size=args.batch_size, num_workers=args.num_workers, seed=iFold,
                                             task=args.setting["task"], size=args.size,
                                             batch_size_test=args.batch_size_test)

        # Set adapter
        adapter = init_adapter(model, args)

        # Fit adapter
        adapter.fit(args.loaders)

        # Test model - predict and evaluate
        if args.loaders["test"] is not None:
            refs, preds = adapter.predict(args.loaders["test"])
            metrics_fold = evaluate(refs, preds, args.setting["task"])
            args.metrics_test.append(metrics_fold)

        # Store weights
        args.weights.append(adapter.model.state_dict())

        # External testing for OOD
        if args.experiment_test[0] != "":
            for i_external in range(len(args.experiment_test)):
                print("External testing: " + args.experiment_test[i_external])

                # Get setting
                setting_external = get_experiment_setting(args.experiment_test[i_external])

                # Prepare dataloaders
                loaders_external = get_dataloader_splits(setting_external["dataframe"],
                                                         args.data_root_path + args.setting["base_samples_path"],
                                                         args.setting["targets"], shots_train="0%", shots_val="0%",
                                                         shots_test="100%", balance=False,
                                                         batch_size=args.batch_size_test,
                                                         batch_size_test=args.batch_size_test,
                                                         num_workers=args.num_workers, seed=iFold,
                                                         task=args.setting["task"], size=args.size)
                # Test model - predict and evaluate
                refs, preds = adapter.predict(loaders_external["test"])
                metrics = evaluate(refs, preds, args.setting["task"])
                args.metrics_external[i_external].append(metrics)

    # Get metrics averaged across folds
    if args.loaders["test"] is not None:
        print("\nTransferability (cross-validation)", end="\n")
        args.metrics = average_folds_results(args.metrics_test, args.setting["task"])
    else:
        args.metrics = None

    # Save experiment metrics
    save_results(args.metrics, args.out_path, id_experiment=generate_experiment_id(args),
                 id_metrics="metrics", save_model=args.save_model, weights=args.weights)

    # Get metrics averaged across fold for external testing
    if args.experiment_test[0] != "":
        for i_external in range(len(args.experiment_test)):
            print("External testing: " + args.experiment_test[i_external])
            metrics = average_folds_results(args.metrics_external[i_external], args.setting["task"])
            # Save experiment metrics
            save_results(metrics, args.out_path, id_experiment=generate_experiment_id(args),
                         id_metrics=args.experiment_test[i_external], save_model=False)


def main():
    parser = argparse.ArgumentParser()

    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_TRANSFERABILITY, help='output path')
    parser.add_argument('--save_model', default=False, type=lambda x: (str(x).lower() == 'true'))

    # Experiment
    parser.add_argument('--experiment', default='chexpert_5x200',
                        help='chexpert_5x200 - mimic_5x200 - covid_train - rsna_pneumonia_train - mimic_pneumonia')
    parser.add_argument('--experiment_test', default='',
                        help='rsna_pneumonia_test, covid_test',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--method', default='zero_shot',
                        help='lp - tipAdapter - tipAdapter-f - clipAdapter'
                             'FT - FT_last - LP_FT -LP_FT_bn_last - FT_freeze_all'
                             'zero_shot -')

    # Model base weights and architecture
    parser.add_argument('--weights_path', default=None,
                        help='./local_data/results/pretraining/resnet_v2_epoch5.pth')
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--init_imagenet', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--architecture', default='resnet_v2', help='resnet_v2 -- efficientnet')
    parser.add_argument('--project_features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--norm_features', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Dataloaders: Training Validation - Testing
    parser.add_argument('--shots_train', default="80%", type=lambda x: (str(x)))
    parser.add_argument('--shots_val', default="0%", type=lambda x: (str(x)))
    parser.add_argument('--shots_test', default="20%", type=lambda x: (str(x)))
    parser.add_argument('--balance', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--folds', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--batch_size_test', default=4, type=int)
    parser.add_argument('--size', default=(224, 224), help="(512, 512) | (2048, 4096) ")

    # Vision adapters setting
    parser.add_argument('--ensemble', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Adapters augmentation strategies
    parser.add_argument('--fta', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--tta', default=False, type=lambda x: (str(x).lower() == 'true'))

    # Fine tuning setting
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--update_bn', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--freeze_classifier', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--last_lp', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_best', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--patience', default=10, type=int)

    # Saving test predictions option
    parser.add_argument('--test_from_folder', default=[], type=list)

    # Resources
    parser.add_argument('--num_workers', default=0, type=int, help='workers number for DataLoader')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()