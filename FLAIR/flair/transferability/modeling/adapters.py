"""
This script contains adapters for fast adaptation of
FLAIR modelo to downstream tasks/domains.

In particular, these adapters work over the vision and text
embeddings. Also, this code contains a Wrapper for zero-shot
classification

Implemented adapters:
Zero-shot, Linear Probe (LP), ClipAdapter, TipAdapter, TipAdapter-f
"""

import copy
import random
import torch
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from flair.pretraining.data.transforms import augmentations_pretraining

from utils import *
from torch.autograd import Variable

import more_itertools 

# for coop
from torch.optim.lr_scheduler import _LRScheduler
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from transformers import AutoModel, AutoTokenizer, logging

# for cocoop
from collections import OrderedDict

# for prograd
from torch.nn.modules.loss import _Loss


# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
The first section contains only-vision adapters (i.e. linear probing)
"""


class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        # Set model and number of targets
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.num_targets = len(targets)
        # Augmentation for training and for test-time augmentation
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20

    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()

        epoch_iterator = tqdm(
            data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True
        )

        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision encoder
                x = self.model.vision_model(images)

            X.extend(x.cpu().detach().numpy())
            Y.extend(batch["label"].numpy())

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def fit(self, loaders, transforms=None):
        data_loader = loaders["train"]
        data_loader_val = loaders["val"]

        if self.fta:
            transforms = augmentations_pretraining

        # Extract features and labels from generator
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)
            X_val, Y_val = self.extract_vision_features(data_loader_val, transforms=transforms)

        # Perform logistic regression
        # self.train(X, Y)
        self.train(X, Y, X_val, Y_val)

    def train(self, X, Y):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return

    def predict(self, loader, transforms=None):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return


class LinearProbe(AdapterWrapper):
    def __init__(self, model, targets, tta=False, fta=False, c=0.316):
        super().__init__(model, targets, tta=tta, fta=fta)
        # self.classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0,
        #                                      class_weight="balanced")                
        self.num_step = 8
    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        X_val = torch.tensor(X_val).to(device)
        Y_val = torch.tensor(Y_val).to(device)

        # # Train classifier
        # self.classifier.fit(X, Y)

        # search initialization
        search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        acc_list = []
        for c_weight in search_list:
            clf = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_weight).fit(X.cpu().numpy(), Y.cpu().numpy())
            pred = clf.predict(X_val.cpu().numpy())
            acc_val = sum(pred == Y_val.cpu().numpy()) / len(Y_val.cpu().numpy())
            acc_list.append(acc_val)

        # print(acc_list, flush=True)

        # binary search
        peak_idx = np.argmax(acc_list)
        c_peak = search_list[peak_idx]
        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak

        test_acc_step_list = np.zeros([self.num_step])

        def binary_search(c_left, c_right, step):
            clf_left = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_left).fit(X.cpu().numpy(), Y.cpu().numpy())
            pred_left = clf_left.predict(X_val.cpu().numpy())
            acc_left = sum(pred_left == Y_val.cpu().numpy()) / len(Y_val.cpu().numpy())
            print("Val accuracy (Left): {:.2f}".format(100 * acc_left), flush=True)

            clf_right = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_right).fit(X.cpu().numpy(), Y.cpu().numpy())
            pred_right = clf_right.predict(X_val.cpu().numpy())
            acc_right = sum(pred_right == Y_val.cpu().numpy()) / len(Y_val.cpu().numpy())
            print("Val accuracy (Right): {:.2f}".format(100 * acc_right), flush=True)

            # find maximum and update ranges
            if acc_left < acc_right:
                # c_final = c_right
                self.clf_final = copy.deepcopy(clf_right)
                # range for the next step
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                # c_final = c_left
                self.clf_final = copy.deepcopy(clf_left)
                # range for the next step
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)

            # pred = clf_final.predict(test_features.cpu().numpy())
            # test_acc = 100 * sum(pred == test_labels.cpu().numpy()) / len(pred)
            # print("Test Accuracy: {:.2f}".format(test_acc), flush=True)
            # test_acc_step_list[step] = test_acc
            return np.power(10, c_left), np.power(10, c_right), step#, test_acc_step_list

        for step in range(self.num_step):
            print('step is ',step)
            c_left, c_right, step = binary_search(c_left, c_right, step)
        # # save results of last step
        # acc_test_final = test_acc_step_list[-1]

        # Set Linear Probe classifier into FLAIR model
        self.model.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True)
        self.model.classifier.weight = torch.nn.Parameter(torch.tensor(self.clf_final.coef_).to(torch.float32))
        self.model.classifier.bias = torch.nn.Parameter(torch.tensor(self.clf_final.intercept_).to(torch.float32))
        self.model.classifier.to(device)

    def predict(self, loader, transforms=None):

        self.model.eval()

        # Set transforms on test-time augmentation
        if self.tta:
            transforms = augmentations_pretraining

        epoch_iterator = tqdm(
            loader, desc="Predicting (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        with torch.no_grad():
            refs, preds = [], []
            for step, batch in enumerate(epoch_iterator):
                images = batch["image"].to(device).to(torch.float32)
                Y = batch["label"].to(device).to(torch.long)

                # Forward
                if self.tta:
                    preds_tta = []
                    for i in range(self.number_augmentations):
                        x = self.model.vision_model(transforms(images))
                        score = self.model.classifier(x)
                        preds_tta.append(score.unsqueeze(-1))
                    score = torch.concat(preds_tta, -1).mean(-1)
                else:
                    x = self.model.vision_model(images)
                    score = self.model.classifier(x)
                # Activation for prediction
                if score.shape[-1] == 1:  # Binary case
                    score = torch.sigmoid(score)
                    score = torch.concat([1 - score, score], -1)
                else:  # Multi-class case
                    score = torch.softmax(score, -1)
                torch.cuda.empty_cache()

                refs.append(Y.cpu().detach().numpy())
                preds.append(score.cpu().detach().numpy())

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds


"""
This section contains multimodal (vision-language) adapters.
"""


class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # Compute text prototypes
        self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()),
                                                                                domain_knowledge=domain_knowledge)


class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    def fit(self, loaders, transforms=None):
        """
        No training in zero-shot prediction
        """
        return

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = 100*torch.matmul(torch.tensor(X), self.text_embeds.t()) #* self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = 100*torch.matmul(X, self.text_embeds.t().to(device)) #* self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        return refs, preds


class ClipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim
        self.reduction = 4
        self.WARMUP_EPOCH = 1
        self.WARMUP_CONS_LR = 0.00001
        self.batch_size = 2 # 2 for refuge, 32 for others
        self.lr = 0.02 # 0.01
        self.epochs = 200 #200
        self.alpha = 0.5

        # self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
        #                                    torch.nn.ReLU(inplace=True),
        #                                    torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
        #                                    torch.nn.ReLU(inplace=True)).to(device)

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    # Compute residual CLIP-Adapter
                    X = self.residual_adapter(X, self.alpha, self.adapter)
                    # Compute similarity
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                # Compute residual CLIP-Adapter
                X = self.residual_adapter(X, self.alpha, self.adapter)
                # Compute similarity
                score = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        X_val = torch.tensor(X_val).to(device)
        Y_val = torch.tensor(Y_val).to(device)

        # optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])

        # Set adapter
        adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True)).to(device)

        optimizer = torch.optim.SGD(adapter.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, self.WARMUP_EPOCH,
                self.WARMUP_CONS_LR
            )
        # Train
        print('\nStart Training procedure')
           
        best_acc, best_epoch = 0.0, 0
        # indexes = np.arange(0, X.shape[0])
        # random.shuffle(indexes)
        indexes = [i for i in range(0, X.shape[0])]
        random.shuffle(indexes)
        for i_epoch in range(self.epochs):
            # loss_epoch = 0.0
            # Train
            adapter.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(i_epoch, self.epochs))
            list_batch = list(more_itertools.chunked(indexes, self.batch_size))
            for i_sample in range(len(list_batch)):
                # X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)
                # target = Y[indexes[i_sample]].unsqueeze(0).to(device)
                X_batch = X[list_batch[i_sample], :].to(device)
                # print('the size of X_batch is ',X_batch.shape)
                target = Y[list_batch[i_sample]].to(device)

                # Compute residual CLIP-Adapter
                X_batch = self.residual_adapter(X_batch, self.alpha, adapter)
                # X_res = adapter(X_batch)
                # X_batch = self.alpha * X_res + (1 - self.alpha) * X_batch

                # Compute logits
                logits = self.model.logit_scale.exp() * X_batch @ self.text_embeds.t().to(device)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, target)

                # acc = cls_acc(logits, target)
                acc = np.mean(logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            #     loss_epoch += loss.item()/X.shape[0]

            # print('loss=%2.5f' % loss_epoch, end="\n")
            adapter.eval()
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            # Compute residual CLIP-Adapter
            X_val_batch = self.residual_adapter(X_val, self.alpha, adapter)
            # X_val_res = adapter(X_val)
            # X_val_batch = self.alpha * X_val_res + (1 - self.alpha) * X_val

            # Compute logits
            logits_val = self.model.logit_scale.exp() * X_val_batch @ self.text_embeds.t().to(device)


            # acc = cls_acc(logits, val_labels)
            acc = np.mean(logits_val.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0
            
            print("**** Clip-Adapter's val accuracy: {:.4f}. ****\n".format(acc))
            if acc > best_acc:
                print('the adapter is changed!')
                best_acc = acc
                best_epoch = i_epoch
                # Storage trained adapter
                self.adapter = copy.deepcopy(adapter)


    def residual_adapter(self, X, alpha, adapter):
        # Compute residual CLIP-Adapter
        X_res = adapter(X)
        X = alpha * X_res + (1 - alpha) * X
        X = X / X.norm(dim=-1, keepdim=True)
        return X

class TipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.train_tip = True

        # Init cache values
        self.cache_keys = []
        self.cache_values = []
        self.adapter_layer = []
        self.lr = 0.001
        self.epochs = 20
        self.init_alpha_scale = 10
        self.init_beta = 1
        self.init_alpha = 1
        self.search_scale = [50,50]
        self.search_step = [200,200]
        self.batch_size = 32

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.adapter(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        X_val = torch.tensor(X_val).to(device)
        Y_val = torch.tensor(Y_val).to(device)

        self.cache_keys = torch.transpose(X, 1, 0).to(torch.float32).to(device)
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device)

        beta, alpha = self.init_beta, self.init_alpha

        if self.train_tip:

            # Enable the cached keys to be learnable
            adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
            adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())
            adapter_layer = adapter_layer.to(device)

            optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=self.lr, eps=1e-4)
        
            indexes = [i for i in range(0, X.shape[0])]
            random.shuffle(indexes)
            list_batch = list(more_itertools.chunked(indexes, self.batch_size))

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * len(list_batch))

            print('alpha is ', alpha)
            print('beta is ', beta)

            # Training Prodecure
            print("**** Start Training **** \n")
            best_acc, best_epoch = 0.0, 0
            for i_epoch in range(self.epochs):
                # loss_epoch = 0.0
                # Train
                adapter_layer.train()
                correct_samples, all_samples = 0, 0
                loss_list = []
                print('Train Epoch: {:} / {:}'.format(i_epoch, self.epochs))
                for i_sample in range(len(list_batch)):
                    # X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)
                    # target = Y[indexes[i_sample]].unsqueeze(0).to(device)
                    image = self.cache_keys[:, list_batch[i_sample]].to(device).t()
                    target = Y[list_batch[i_sample]].to(device)

                    # Zero-shot CLIP
                    # print('the size of image is ',image.shape)
                    # print('the size of self.text_embeds is ',self.text_embeds.shape)
                    # print('the size of target is ',target.shape)
                    clip_logits = 100. * (image @ self.text_embeds.t())

                    # Tip-Adapter
                    affinity = adapter_layer(image)
                    cache_logits = torch.exp(((-1) * (beta - beta * affinity))) @ self.cache_values
                    # cache_logits /= X.shape[0]
                    # cache_logits *= self.model.logit_scale.exp()

                    tip_logits = clip_logits + cache_logits * alpha

                    loss = torch.nn.functional.cross_entropy(tip_logits, target)

                    acc = np.mean(tip_logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                    correct_samples += acc / 100 * len(tip_logits)
                    all_samples += len(tip_logits)
                    loss_list.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                current_lr = scheduler.get_last_lr()[0]
                print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

                # Eval
                adapter_layer.eval()

                affinity_val = adapter_layer(X_val)
                cache_logits_val = torch.exp(((-1) * (beta - beta * affinity_val))) @ self.cache_values

                clip_logits_val = 100. * (X_val @ self.text_embeds.t())
                tip_logits_val = clip_logits_val + cache_logits_val * alpha
                acc = np.mean(tip_logits_val.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0

                # print('adapter_layer is ',adapter_layer)

                print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(acc))
                if acc > best_acc:
                    # print('the best acc is ',best_acc)
                    best_acc = acc
                    best_epoch = i_epoch
                    # Storage trained adapter
                    self.adapter_layer = copy.deepcopy(adapter_layer)
            
            # print('before search, self.adapter_layer is ',self.adapter_layer)
            # Search Hyperparameters
            self.beta, self.alpha = self.search_hp_tip(X_val, Y_val, adapter=self.adapter_layer)
            # print('after search, self.adapter_layer is ',self.adapter_layer)

    def search_init_hp(self, alpha, beta, X_val, Y_val):
        adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
        adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())
        adapter_layer = adapter_layer.to(device)

        optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=self.lr, eps=1e-4)

        indexes = [i for i in range(0, X_val.shape[0])]
        random.shuffle(indexes)
        list_batch = list(more_itertools.chunked(indexes, self.batch_size))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * len(list_batch))

        for i_epoch in range(self.epochs):
            # loss_epoch = 0.0
            # Train
            adapter_layer.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Val Epoch: {:} / {:}'.format(i_epoch, self.epochs))
            for i_sample in range(len(list_batch)):
                X_batch = X_val[list_batch[i_sample], :].to(device)
                target = Y_val[list_batch[i_sample]].to(device)

                # Zero-shot CLIP
                clip_logits = 100. * (X_batch @ self.text_embeds.t())

                # Tip-Adapter
                affinity = adapter_layer(X_batch)
                cache_logits = torch.exp(((-1) * (beta - beta * affinity))) @ self.cache_values
                # cache_logits /= X.shape[0]
                # cache_logits *= self.model.logit_scale.exp()

                tip_logits = clip_logits + cache_logits * alpha

                loss = torch.nn.functional.cross_entropy(tip_logits, target)

                acc = np.mean(tip_logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            # Eval
            adapter_layer.eval()

        return adapter_layer

    def search_hp_tip(self, X_val, Y_val, adapter=None):

        beta_list = [i * (self.search_scale[0] - 0.1) / self.search_step[0] + 0.1 for i in range(self.search_step[0])]
        alpha_list = [i * (self.search_scale[1] - 0.1) / self.search_step[1] + 0.1 for i in range(self.search_step[1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(X_val)
                else:
                    affinity = X_val @ self.cache_keys

                cache_logits = torch.exp(((-1) * (beta - beta * affinity))) @ self.cache_values
                clip_logits = 100. * X_val @ self.text_embeds.t()
                tip_logits = clip_logits + cache_logits * alpha
                acc = np.mean(tip_logits.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

        return best_beta, best_alpha

    def adapter(self, X):
        print('the size of X is ',X.shape)
        print('self.adapter_layer is ',self.adapter_layer)
        # Zero-shot CLIP
        clip_logits = 100. * (X @ self.text_embeds.t())

        # Tip-Adapter
        if not self.train_tip:
            affinity = X @ self.cache_keys
        else:
            affinity = self.adapter_layer(X)

        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values
        logits = clip_logits + cache_logits * self.alpha

        return logits


class TipAdapter_f(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.train_tip = True

        # Init cache values
        self.cache_keys = []
        self.cache_values = []
        self.adapter_layer = []
        self.lr = 0.001
        self.epochs = 20
        self.init_alpha_scale = 10
        self.init_beta = 1
        self.init_alpha = 1
        self.search_scale = [50,50]
        self.search_step = [200,200]
        self.batch_size = 32

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.adapter(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        X_val = torch.tensor(X_val).to(device)
        Y_val = torch.tensor(Y_val).to(device)

        self.cache_keys = torch.transpose(X, 1, 0).to(torch.float32).to(device)
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device)

        beta, alpha = self.init_beta, self.init_alpha

        if self.train_tip:

            # Enable the cached keys to be learnable
            adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
            adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())
            adapter_layer = adapter_layer.to(device)

            optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=self.lr, eps=1e-4)
        
            indexes = [i for i in range(0, X.shape[0])]
            random.shuffle(indexes)
            list_batch = list(more_itertools.chunked(indexes, self.batch_size))

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * len(list_batch))

            # alpha initialization
            best_acc = 0.0
            print("**** Searching for best initialization of alpha **** \n")
            for init_alpha in range(1,11):
                init_adapter = self.search_init_hp(init_alpha, beta, X_val, Y_val)
                affinity = init_adapter(X_val)
                cache_logits = torch.exp(((-1) * (beta - beta * affinity))) @ self.cache_values
                clip_logits = 100. * (X_val @ self.text_embeds.t())
                tip_logits = clip_logits + cache_logits * init_alpha
                acc = np.mean(tip_logits.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0
                if acc > best_acc:
                    best_acc = acc
                    alpha = init_alpha
                    adapter_layer = init_adapter
            print('alpha is ', alpha)
            print('beta is ', beta)

            # Training Prodecure
            print("**** Start Training **** \n")
            best_acc, best_epoch = 0.0, 0
            for i_epoch in range(self.epochs):
                # loss_epoch = 0.0
                # Train
                adapter_layer.train()
                correct_samples, all_samples = 0, 0
                loss_list = []
                print('Train Epoch: {:} / {:}'.format(i_epoch, self.epochs))
                for i_sample in range(len(list_batch)):
                    # X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)
                    # target = Y[indexes[i_sample]].unsqueeze(0).to(device)
                    image = self.cache_keys[:, list_batch[i_sample]].to(device).t()
                    target = Y[list_batch[i_sample]].to(device)

                    # Zero-shot CLIP
                    # print('the size of image is ',image.shape)
                    # print('the size of self.text_embeds is ',self.text_embeds.shape)
                    # print('the size of target is ',target.shape)
                    clip_logits = 100. * (image @ self.text_embeds.t())

                    # Tip-Adapter
                    affinity = adapter_layer(image)
                    cache_logits = torch.exp(((-1) * (beta - beta * affinity))) @ self.cache_values
                    # cache_logits /= X.shape[0]
                    # cache_logits *= self.model.logit_scale.exp()

                    tip_logits = clip_logits + cache_logits * alpha

                    loss = torch.nn.functional.cross_entropy(tip_logits, target)

                    acc = np.mean(tip_logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                    correct_samples += acc / 100 * len(tip_logits)
                    all_samples += len(tip_logits)
                    loss_list.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                current_lr = scheduler.get_last_lr()[0]
                print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

                # Eval
                adapter_layer.eval()

                affinity_val = adapter_layer(X_val)
                cache_logits_val = torch.exp(((-1) * (beta - beta * affinity_val))) @ self.cache_values

                clip_logits_val = 100. * (X_val @ self.text_embeds.t())
                tip_logits_val = clip_logits_val + cache_logits_val * alpha
                acc = np.mean(tip_logits_val.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0

                # print('adapter_layer is ',adapter_layer)

                print("**** Tip-Adapter-F's val accuracy: {:.2f}. ****\n".format(acc))
                if acc > best_acc:
                    # print('the best acc is ',best_acc)
                    best_acc = acc
                    best_epoch = i_epoch
                    # Storage trained adapter
                    self.adapter_layer = copy.deepcopy(adapter_layer)
            
            # print('before search, self.adapter_layer is ',self.adapter_layer)
            # Search Hyperparameters
            self.beta, self.alpha = self.search_hp_tip(X_val, Y_val, adapter=self.adapter_layer)
            # print('after search, self.adapter_layer is ',self.adapter_layer)

    def search_init_hp(self, alpha, beta, X_val, Y_val):
        adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
        adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())
        adapter_layer = adapter_layer.to(device)

        optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=self.lr, eps=1e-4)

        indexes = [i for i in range(0, X_val.shape[0])]
        random.shuffle(indexes)
        list_batch = list(more_itertools.chunked(indexes, self.batch_size))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs * len(list_batch))

        for i_epoch in range(self.epochs):
            # loss_epoch = 0.0
            # Train
            adapter_layer.train()
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Val Epoch: {:} / {:}'.format(i_epoch, self.epochs))
            for i_sample in range(len(list_batch)):
                X_batch = X_val[list_batch[i_sample], :].to(device)
                target = Y_val[list_batch[i_sample]].to(device)

                # Zero-shot CLIP
                clip_logits = 100. * (X_batch @ self.text_embeds.t())

                # Tip-Adapter
                affinity = adapter_layer(X_batch)
                cache_logits = torch.exp(((-1) * (beta - beta * affinity))) @ self.cache_values
                # cache_logits /= X.shape[0]
                # cache_logits *= self.model.logit_scale.exp()

                tip_logits = clip_logits + cache_logits * alpha

                loss = torch.nn.functional.cross_entropy(tip_logits, target)

                acc = np.mean(tip_logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            # Eval
            adapter_layer.eval()

        return adapter_layer

    def search_hp_tip(self, X_val, Y_val, adapter=None):

        beta_list = [i * (self.search_scale[0] - 0.1) / self.search_step[0] + 0.1 for i in range(self.search_step[0])]
        alpha_list = [i * (self.search_scale[1] - 0.1) / self.search_step[1] + 0.1 for i in range(self.search_step[1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(X_val)
                else:
                    affinity = X_val @ self.cache_keys

                cache_logits = torch.exp(((-1) * (beta - beta * affinity))) @ self.cache_values
                clip_logits = 100. * X_val @ self.text_embeds.t()
                tip_logits = clip_logits + cache_logits * alpha
                acc = np.mean(tip_logits.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

        return best_beta, best_alpha

    def adapter(self, X):
        print('the size of X is ',X.shape)
        print('self.adapter_layer is ',self.adapter_layer)
        # Zero-shot CLIP
        clip_logits = 100. * (X @ self.text_embeds.t())

        # Tip-Adapter
        if not self.train_tip:
            affinity = X @ self.cache_keys
        else:
            affinity = self.adapter_layer(X)

        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values
        logits = clip_logits + cache_logits * self.alpha

        return logits


class LinearProbe2(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.lr_alpha = 10
        self.lr_temp0 = 1

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            # this is used!!!
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.adapter(X)


        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        X_val = torch.tensor(X_val)
        Y_val = torch.tensor(Y_val)
        num_class = Y.unique().size(0)
        shot = int(X.shape[0]/num_class)

        # TRAINING
        epochs = 300

        indexes = np.arange(0, X.shape[0])
        random.shuffle(indexes)
        loss_epoch = 0.0
        best_acc = 0.0
        X_batch = X[indexes, :].to(device)
        target = Y[indexes].to(device)
        features = X_batch
        text_weights = self.text_embeds.t()
        val_features = X_val.to(device)
        val_target = Y_val.to(device)

        # print('the size of text_weights is ',text_weights.shape)
        # print('the size of features is ',features.shape)
        # print('the size of target is ',target.shape)
        # print('target is ',target)
        # print('the size of val_features is ',val_features.shape)
        # print('the size of val_target is ',val_target.shape)
        # print('shot is ',shot)


        centroids = compute_centroids(features.unsqueeze(0), target.unsqueeze(0))  # [batch, num_class, d]
    
        classifier = nn.Linear(features.shape[1], int(features.shape[0]/shot),bias=True).to(device).cuda()
        classifier.weight.data = centroids[0]

        # # lr_w
        # best_acc_val = 0
        # for lr_temp0 in range(1,6):
        lr_temp  = np.log(shot + 1) * self.lr_temp0
        # lr_w
        # lr_temp = calculate_lr_w(features)
        # print('lr_temp is ',lr_temp)

        # init_alpha
        final_init_alpha_mean= calculate_init_alpha(features, target, shot, text_weights)

        alpha_vec = Variable(final_init_alpha_mean * torch.ones(1, int(features.shape[0]/shot)).to(device).cuda(), requires_grad=True)

        print('alpha_vec is ',alpha_vec)

        # # lr_alpha
        self.lr_alpha = calculate_lr_alpha(features, text_weights)

        print('self.lr_alpha is ',self.lr_alpha)

        optimizer = torch.optim.SGD(classifier.parameters(), lr_temp, momentum=0.9)


        classifier.train()
        for i_epoch in range(epochs):
            print('Running model for epoch: {}'.format(i_epoch))
            vision_logits = classifier(features)
            text_logits = features @ text_weights

            # Compute logits
            logits = vision_logits + torch.ones(features.shape[0],1).to(device).cuda() @ alpha_vec * text_logits
            acc = np.mean(logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
            print('The accuracy for training data is ',acc)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # update for alpha
            if (i_epoch + 1) % 10 == 0:
                alpha_vec.data -= self.lr_alpha * alpha_vec.grad.data

            classifier.eval()
            vision_logits_val = classifier(val_features)
            text_logits_val = val_features.detach() @ text_weights
            logits_val = vision_logits_val + torch.ones(val_features.shape[0], 1).to(device).cuda() @ alpha_vec * text_logits_val
            acc_val = np.mean(logits_val.argmax(dim=1).cpu().numpy() ==  val_target.cpu().numpy()) * 100.0
            print('The accuracy for val data is ',acc_val)
        
            if acc_val >= best_acc:
                best_acc = acc_val
                best_epoch = i_epoch
                self.best_alpha = copy.deepcopy(alpha_vec.data)
                # # Storage trained adapter
                # torch.save(classifier, '/export/livia/home/vision/Yhuang/FLAIR/save_model/best_val_model.pt')
                self.classifier = copy.deepcopy(classifier) # copy.deepcopy is important!!!
        # if best_acc_val < best_acc:
        #         best_acc_val = best_acc
        #         best_lr_temp0 = lr_temp0
        # print('the optimal lr_temp0 is ',best_lr_temp0)



    def adapter(self, X):
        print('the size of X is ',X.shape)
        # print('loading the best model')
        # classifier = torch.load('/export/livia/home/vision/Yhuang/FLAIR/save_model/best_val_model.pt')
        vision_logits = self.classifier(X)
        text_logits = X.detach() @ self.text_embeds.t()
        logits = vision_logits + torch.ones(X.shape[0], 1).to(device).cuda() @ self.best_alpha * text_logits

        return logits

                
class CoOp(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim
        self.reduction = 4
        self.WARMUP_EPOCH = 1
        self.WARMUP_CONS_LR = 0.00001
        self.batch_size = 32
        self.lr = 0.1
        self.epochs = 200 #1: 50, 2, 4: 100, 8, 16: 200
        self.NCTX = 16
        self.CTX_INIT = ""
        self.CLASS_TOKEN_POSITION = "end"
        self.CSC = False
        self.PREC = "fp16"
        self.backbone = "RN50"
        self.classnames = list(targets.keys())

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.coop_model(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.coop_model(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        print('the size of test data is ',X.shape)

        return refs, preds
        
    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        X_val = torch.tensor(X_val).to(device)
        Y_val = torch.tensor(Y_val).to(device)

        if self.PREC == "fp32" or self.PREC == "amp":
            self.model.float()
        self.model.eval()
        print("Building custom FLAIR")
        coop_model = CustomFLAIR(self.classnames, self.model).cuda()
        # initialize the coop_model
        self.coop_model = copy.deepcopy(coop_model) 
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in coop_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)    

        coop_model.to("cuda")

        # Set adapter
        optimizer = torch.optim.SGD(coop_model.prompt_learner.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, self.WARMUP_EPOCH,
                self.WARMUP_CONS_LR
            )

        self.scaler = GradScaler() if self.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            coop_model = nn.DataParallel(coop_model)

        # Train
        print('\nStart Training procedure')
           
        best_acc, best_epoch = 0.0, 0
        # indexes = np.arange(0, X.shape[0])
        # random.shuffle(indexes)
        indexes = [i for i in range(0, X.shape[0])]
        random.shuffle(indexes)
        print('epoch is ', self.epochs)
        for i_epoch in range(self.epochs):
            # loss_epoch = 0.0
            # Train
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(i_epoch, self.epochs))
            list_batch = list(more_itertools.chunked(indexes, self.batch_size))
            for i_sample in range(len(list_batch)):
                X_batch = X[list_batch[i_sample], :].to(device)
                target = Y[list_batch[i_sample]].to(device)

                # Compute logits
                logits = coop_model(X_batch)
                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, target)

                # acc = cls_acc(logits, target)
                acc = np.mean(logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


            # print('loss=%2.5f' % loss_epoch, end="\n")
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            coop_model.eval()

            # Compute logits
            logits_val = coop_model(X_val)

            # acc = cls_acc(logits, val_labels)
            acc = np.mean(logits_val.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0
            
            print("**** CoOp's val accuracy: {:.4f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = i_epoch
                # Storage trained adapter
                self.coop_model = copy.deepcopy(coop_model)

class TextEncoder(nn.Module):
    def __init__(self, flair_model):
        super().__init__()
        self.flair_model = flair_model
    
    def forward(self, input_ids, attention_mask):
        # Forward vision and text encoder
        with torch.no_grad():
            text_embeds = self.text_model(input_ids, attention_mask)

        return text_embeds

    
class PromptLearner(nn.Module):
    def __init__(self, classnames, flair_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        CSC = False
        CLASS_TOKEN_POSITION = "end"
        # ctx_dim = flair_model.ln_final.weight.shape[0]
        ctx_dim = 768
        dtype = torch.float32
        bert_type='emilyalsentzer/Bio_ClinicalBERT'

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            clip_model.to("cpu")
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print('the number of parameters for CoOp is ',self.ctx.shape)

        classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        text_token = flair_model.text_model.tokenizer(prompts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            input_ids = text_token["input_ids"].to(device).to(torch.long)
            attention_mask = text_token["attention_mask"].to(device).to(torch.long) # keeps the same
            # Forwards trough text encoder
            model_tempt = AutoModel.from_pretrained(bert_type, output_hidden_states=True).cuda()
            output = model_tempt(input_ids=input_ids, attention_mask=attention_mask)

            # Combine last feature layers to compute text embedding
            last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                            output['hidden_states'][-1]]) # 3*5*28*768
            last_hidden_states_temp = last_hidden_states.permute(1, 0, 2, 3).mean(1) #5*28*768

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", last_hidden_states_temp[:, :1, :])  
        self.register_buffer("token_suffix", last_hidden_states_temp[:, 1 + n_ctx :, :])  

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            last_hidden_states_temp = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return last_hidden_states_temp
    
class CustomFLAIR(nn.Module):
    def __init__(self, classnames, flair_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, flair_model)
        self.flair_model = flair_model
        self.logit_scale = flair_model.logit_scale
        self.proj_dim = 512
        self.proj_bias = False
        self.projection = True
        self.norm = True
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, self.proj_dim, bias=self.proj_bias),
                                                    projection=self.projection, norm=self.norm)

    def forward(self, image_features):

        last_hidden_states_temp = self.prompt_learner()
        embed = last_hidden_states_temp.mean(1)

        # Compute projection from text embedding to multi-modal projection
        text_embeds = self.projection_head_text(embed)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_embeds.t()

        return logits
    

class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)
                
class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):

        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x

    
class CoCoOp(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim
        self.reduction = 4
        self.WARMUP_EPOCH = 1
        self.WARMUP_CONS_LR = 0.00001
        self.batch_size = 1
        self.lr = 2.0
        self.epochs = 200 #1: 50, 2, 4: 100, 8, 16: 200
        self.CSC = False
        self.PREC = "fp16"
        self.classnames = list(targets.keys())

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.cocoop_model(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.cocoop_model(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        print('the size of test data is ',X.shape)

        return refs, preds
        
    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        X_val = torch.tensor(X_val).to(device)
        Y_val = torch.tensor(Y_val).to(device)

        if self.PREC == "fp32" or self.PREC == "amp":
            self.model.float()
        self.model.eval()
        print("Building custom FLAIR")
        cocoop_model = CustomFLAIR_cocoop(self.classnames, self.model).cuda()
        # initialize the coop_model
        self.cocoop_model = copy.deepcopy(cocoop_model) 
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in cocoop_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)    

        cocoop_model.to("cuda")

        # Set adapter
        optimizer = torch.optim.SGD(cocoop_model.prompt_learner.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, self.WARMUP_EPOCH,
                self.WARMUP_CONS_LR
            )

        self.scaler = GradScaler() if self.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            cocoop_model = nn.DataParallel(cocoop_model)

        # Train
        print('\nStart Training procedure')
           
        best_acc, best_epoch = 0.0, 0
        # indexes = np.arange(0, X.shape[0])
        # random.shuffle(indexes)
        indexes = [i for i in range(0, X.shape[0])]
        random.shuffle(indexes)
        print('epoch is ', self.epochs)
        for i_epoch in range(self.epochs):
            # loss_epoch = 0.0
            # Train
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(i_epoch, self.epochs))
            list_batch = list(more_itertools.chunked(indexes, self.batch_size))
            for i_sample in range(len(list_batch)):
                X_batch = X[list_batch[i_sample], :].to(device)
                target = Y[list_batch[i_sample]].to(device)
                # print('the size of X_batch is ',X_batch.shape)
                # print('the size of target is ',target.shape)

                # Compute logits
                logits = cocoop_model(X_batch)
                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, target)

                # acc = cls_acc(logits, target)
                acc = np.mean(logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


            # print('loss=%2.5f' % loss_epoch, end="\n")
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            cocoop_model.eval()

            # Compute logits
            logits_val = cocoop_model(X_val)

            # acc = cls_acc(logits, val_labels)
            acc = np.mean(logits_val.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0
            
            print("**** CoCoOp's val accuracy: {:.4f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = i_epoch
                # Storage trained adapter
                self.cocoop_model = copy.deepcopy(cocoop_model)

    
class PromptLearner_cocoop(nn.Module):
    def __init__(self, classnames, flair_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = ""
        # ctx_init = "A fundus image of "
        CSC = False
        # ctx_dim = flair_model.ln_final.weight.shape[0]
        ctx_dim = 768
        vis_dim = 512 # need to be changed if necessary!
        dtype = torch.float32
        bert_type='emilyalsentzer/Bio_ClinicalBERT'

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))

            text_token = flair_model.text_model.tokenizer(ctx_init, truncation=True, padding=True, return_tensors='pt')
            with torch.no_grad():
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long) # keeps the same
                # Forwards trough text encoder
                model_tempt = AutoModel.from_pretrained(bert_type, output_hidden_states=True).cuda()
                output = model_tempt(input_ids=input_ids, attention_mask=attention_mask)

                # Combine last feature layers to compute text embedding
                last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                                output['hidden_states'][-1]]) # 3*5*28*768
                last_hidden_states_temp = last_hidden_states.permute(1, 0, 2, 3).mean(1) #5*28*768
                ctx_vectors = last_hidden_states_temp[0, 1 : 1 + n_ctx, :]
                prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        for parameter in self.meta_net.parameters():
            print('the shape of parameter is ',parameter)

        classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        text_token = flair_model.text_model.tokenizer(prompts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            input_ids = text_token["input_ids"].to(device).to(torch.long)
            attention_mask = text_token["attention_mask"].to(device).to(torch.long) # keeps the same
            # Forwards trough text encoder
            model_tempt = AutoModel.from_pretrained(bert_type, output_hidden_states=True).cuda()
            output = model_tempt(input_ids=input_ids, attention_mask=attention_mask)

            # Combine last feature layers to compute text embedding
            last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                            output['hidden_states'][-1]]) # 3*5*28*768
            last_hidden_states_temp = last_hidden_states.permute(1, 0, 2, 3).mean(1) #5*28*768

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", last_hidden_states_temp[:, :1, :])  
        self.register_buffer("token_suffix", last_hidden_states_temp[:, 1 + n_ctx :, :])  

        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        ctx = self.ctx

        prefix = self.token_prefix
        suffix = self.token_suffix
        
        # print('the size of prefix is ',prefix.shape)
        # print('the size of suffix is ',suffix.shape)

        # print('the size of im_features is ',im_features.shape)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        # print('the size of ctx is ',ctx.shape)
        # print('the size of bias is ',bias.shape)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        # print('the size of ctx_shifted is ',ctx_shifted.shape)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            # print('self.n_cls is ',self.n_cls)
            # ctx_i = ctx_shifted_i.expand(self.n_cls, -1, -1)
            # print('the size of ctx_i is ',ctx_i.shape)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts
    
class CustomFLAIR_cocoop(nn.Module):
    def __init__(self, classnames, flair_model):
        super().__init__()
        self.prompt_learner = PromptLearner_cocoop(classnames, flair_model)
        self.flair_model = flair_model
        self.logit_scale = flair_model.logit_scale
        self.proj_dim = 512
        self.proj_bias = False
        self.projection = True
        self.norm = True
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, self.proj_dim, bias=self.proj_bias),
                                                    projection=self.projection, norm=self.norm)

    def forward(self, image_features):

        last_hidden_states_temp = self.prompt_learner(image_features)
        # print('the size of last_hidden_states_temp is ',last_hidden_states_temp.shape)

        logit_scale = self.logit_scale.exp()

        logits = []
        for pts_i, imf_i in zip(last_hidden_states_temp, image_features):
            # print('the size of pts_i is ',pts_i.shape)
            # print('the size of imf_i is ',imf_i.shape)
            embed = pts_i.mean(1)
            # print('the size of embed is ',embed.shape)
            # Compute projection from text embedding to multi-modal projection
            text_embeds = self.projection_head_text(embed)
            l_i = logit_scale * imf_i @ text_embeds.t()
            # print('the size of l_i is ',l_i.shape)
            logits.append(l_i)
        logits = torch.stack(logits)
        # print('the size of logits is ',logits.shape)
        
        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label)
        
        return logits
    

class ProGrad(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim
        self.reduction = 4
        self.WARMUP_EPOCH = 1
        self.WARMUP_CONS_LR = 0.00001
        self.batch_size = 32
        self.lr = 0.1
        self.epochs = 200 #1: 50, 2, 4: 100, 8, 16: 200
        self.NCTX = 16
        self.CTX_INIT = ""
        self.CLASS_TOKEN_POSITION = "end"
        self.CSC = False
        self.PREC = "fp16"
        self.backbone = "RN50"
        self.classnames = list(targets.keys())

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.prograd_model(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.prograd_model(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        print('the size of test data is ',X.shape)

        return refs, preds
        
    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        X_val = torch.tensor(X_val).to(device)
        Y_val = torch.tensor(Y_val).to(device)

        if self.PREC == "fp32" or self.PREC == "amp":
            self.model.float()
        self.model.eval()
        print("Building custom FLAIR")
        prograd_model = CustomFLAIR(self.classnames, self.model).cuda()
        # initialize the prograd_model
        self.prograd_model = copy.deepcopy(prograd_model) 
        
        print("Turning off gradients in both the image and the text encoder")
        names = []
        for name, param in prograd_model.named_parameters():
            names.append(name)
            if "prompt_learner" not in name:
                print('name is ',name)
                param.requires_grad_(False)    

        prograd_model.to("cuda")

        # Set adapter
        optimizer = torch.optim.SGD(prograd_model.prompt_learner.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, self.WARMUP_EPOCH,
                self.WARMUP_CONS_LR
            )

        criterion = ProGradLoss(T=1.0) 

        self.scaler = GradScaler() if self.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            prograd_model = nn.DataParallel(prograd_model)

        # Train
        print('\nStart Training procedure')
           
        best_acc, best_epoch = 0.0, 0
        # indexes = np.arange(0, X.shape[0])
        # random.shuffle(indexes)
        indexes = [i for i in range(0, X.shape[0])]
        random.shuffle(indexes)
        print('epoch is ', self.epochs)
        for i_epoch in range(self.epochs):
            # loss_epoch = 0.0
            # Train
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(i_epoch, self.epochs))
            list_batch = list(more_itertools.chunked(indexes, self.batch_size))
            for i_sample in range(len(list_batch)):
                X_batch = X[list_batch[i_sample], :].to(device)
                target = Y[list_batch[i_sample]].to(device)

                # Compute logits
                logits = prograd_model(X_batch)
                with torch.no_grad():
                    zs_flair_output = 100*torch.matmul(X_batch, self.text_embeds.t())

                # Compute loss
                # loss = torch.nn.functional.cross_entropy(logits, target)
                xe_loss, kl_loss = criterion(logits,
                                              zs_flair_output.detach(),
                                              target)
                
                # acc = cls_acc(logits, target)
                acc = np.mean(logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)


                # backward for ProGrad
                optimizer.zero_grad()
                # backward kl_loss
                kl_loss.backward()
                # normalize gradient
                b_grads = []
                for name in names:
                    for p in prograd_model[name].parameters():
                        b_grads.append(p.grad.clone())
                # optimizer don't step
                for name in names:
                    optimizer[name].zero_grad()

                # backward xe_loss
                xe_loss.backward()
                for name in names:
                    for p, b_grad in zip(prograd_model[name].parameters(), b_grads):
                        # calculate cosine distance
                        b_grad_norm = b_grad / torch.linalg.norm(b_grad)
                        a_grad = p.grad.clone()
                        a_grad_norm = a_grad / torch.linalg.norm(a_grad)

                        if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
                            p.grad = a_grad - lambda_ * torch.dot(
                                a_grad.flatten(), b_grad_norm.flatten()
                            ) * b_grad_norm

                # optimizer
                for name in names:
                    optimizer[name].step()
                scheduler.step()


            # print('loss=%2.5f' % loss_epoch, end="\n")
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            prograd_model.eval()

            # Compute logits
            logits_val = prograd_model(X_val)

            # acc = cls_acc(logits, val_labels)
            acc = np.mean(logits_val.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0
            
            print("**** ProGrad's val accuracy: {:.4f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = i_epoch
                # Storage trained adapter
                self.prograd_model = copy.deepcopy(prograd_model)

    
class PromptLearner_kgcoop(nn.Module):
    def __init__(self, classnames, flair_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        CSC = False
        CLASS_TOKEN_POSITION = "end"
        # ctx_dim = flair_model.ln_final.weight.shape[0]
        ctx_dim = 768
        dtype = torch.float32
        bert_type='emilyalsentzer/Bio_ClinicalBERT'

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            clip_model.to("cpu")
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print('the number of parameters for KgCoOp is ',self.ctx.shape)

        classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        text_token = flair_model.text_model.tokenizer(prompts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            input_ids = text_token["input_ids"].to(device).to(torch.long)
            attention_mask = text_token["attention_mask"].to(device).to(torch.long) # keeps the same
            # Forwards trough text encoder
            model_tempt = AutoModel.from_pretrained(bert_type, output_hidden_states=True).cuda()
            output = model_tempt(input_ids=input_ids, attention_mask=attention_mask)

            # Combine last feature layers to compute text embedding
            last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                            output['hidden_states'][-1]]) # 3*5*28*768
            last_hidden_states_temp = last_hidden_states.permute(1, 0, 2, 3).mean(1) #5*28*768

        temp = "A fundus image of {}."
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")

        text_token_ = flair_model.text_model.tokenizer(prompts_, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            input_ids_ = text_token_["input_ids"].to(device).to(torch.long)
            attention_mask_ = text_token_["attention_mask"].to(device).to(torch.long) # keeps the same
            # Forwards trough text encoder
            model_tempt_ = AutoModel.from_pretrained(bert_type, output_hidden_states=True).cuda()
            output_ = model_tempt_(input_ids=input_ids_, attention_mask=attention_mask_)

            # Combine last feature layers to compute text embedding
            last_hidden_states_ = torch.stack([output_['hidden_states'][1], output_['hidden_states'][2],
                                            output_['hidden_states'][-1]]) # 3*5*28*768
            last_hidden_states_temp_ = last_hidden_states_.permute(1, 0, 2, 3).mean(1) #5*28*768

        self.last_hidden_states_temp_  = last_hidden_states_temp_ 


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", last_hidden_states_temp[:, :1, :])  
        self.register_buffer("token_suffix", last_hidden_states_temp[:, 1 + n_ctx :, :])  

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_token_position = CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            last_hidden_states_temp = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return last_hidden_states_temp
    
class CustomFLAIR(nn.Module):
    def __init__(self, classnames, flair_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, flair_model)
        self.flair_model = flair_model
        self.logit_scale = flair_model.logit_scale
        self.proj_dim = 512
        self.proj_bias = False
        self.projection = True
        self.norm = True
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, self.proj_dim, bias=self.proj_bias),
                                                    projection=self.projection, norm=self.norm)

    def forward(self, image_features):

        last_hidden_states_temp = self.prompt_learner()
        embed = last_hidden_states_temp.mean(1)

        # Compute projection from text embedding to multi-modal projection
        text_embeds = self.projection_head_text(embed)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_embeds.t()

        return logits
    
class ProGradLoss(_Loss):
    def __init__(self, T):
        super(ProGradLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss


class KgCoOp(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim
        self.reduction = 4
        self.WARMUP_EPOCH = 1
        self.WARMUP_CONS_LR = 0.00001
        self.batch_size = 32
        self.lr = 0.5
        self.epochs = 200 #1: 50, 2, 4: 100, 8, 16: 200
        self.NCTX = 16
        self.CTX_INIT = ""
        self.CLASS_TOKEN_POSITION = "end"
        self.CSC = False
        self.PREC = "fp16"
        self.backbone = "RN50"
        self.classnames = list(targets.keys())

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i, _ = self.kgcoop_model(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score, _ = self.kgcoop_model(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        print('the size of test data is ',X.shape)

        return refs, preds
        
    def train(self, X, Y, X_val, Y_val):
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        X_val = torch.tensor(X_val).to(device)
        Y_val = torch.tensor(Y_val).to(device)

        if self.PREC == "fp32" or self.PREC == "amp":
            self.model.float()
        self.model.eval()
        print("Building custom FLAIR")
        kgcoop_model = CustomFLAIR_kgcoop(self.classnames, self.model).cuda()
        # initialize the coop_model
        self.kgcoop_model = copy.deepcopy(kgcoop_model) 
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in kgcoop_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)    

        kgcoop_model.to("cuda")

        # Set adapter
        optimizer = torch.optim.SGD(kgcoop_model.prompt_learner.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, self.WARMUP_EPOCH,
                self.WARMUP_CONS_LR
            )

        self.scaler = GradScaler() if self.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            kgcoop_model = nn.DataParallel(kgcoop_model)

        # Train
        print('\nStart Training procedure')
           
        best_acc, best_epoch = 0.0, 0
        # indexes = np.arange(0, X.shape[0])
        # random.shuffle(indexes)
        indexes = [i for i in range(0, X.shape[0])]
        random.shuffle(indexes)
        print('epoch is ', self.epochs)
        for i_epoch in range(self.epochs):
            # loss_epoch = 0.0
            # Train
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(i_epoch, self.epochs))
            list_batch = list(more_itertools.chunked(indexes, self.batch_size))
            for i_sample in range(len(list_batch)):
                X_batch = X[list_batch[i_sample], :].to(device)
                target = Y[list_batch[i_sample]].to(device)

                # Compute logits
                logits, score = kgcoop_model(X_batch)
                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, target)+8.0*score

                # acc = cls_acc(logits, target)
                acc = np.mean(logits.argmax(dim=1).cpu().numpy() ==  target.cpu().numpy()) * 100.0
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


            # print('loss=%2.5f' % loss_epoch, end="\n")
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

            kgcoop_model.eval()

            # Compute logits
            logits_val, score = kgcoop_model(X_val)

            # acc = cls_acc(logits, val_labels)
            acc = np.mean(logits_val.argmax(dim=1).cpu().numpy() ==  Y_val.cpu().numpy()) * 100.0
            
            print("**** KgCoOp's val accuracy: {:.4f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = i_epoch
                # Storage trained adapter
                self.kgcoop_model = copy.deepcopy(kgcoop_model)

    
class CustomFLAIR_kgcoop(nn.Module):
    def __init__(self, classnames, flair_model):
        super().__init__()
        self.prompt_learner = PromptLearner_kgcoop(classnames, flair_model)
        self.ori_embedding = self.prompt_learner.last_hidden_states_temp_
        self.flair_model = flair_model
        self.logit_scale = flair_model.logit_scale
        self.proj_dim = 512
        self.proj_bias = False
        self.projection = True
        self.norm = True
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, self.proj_dim, bias=self.proj_bias),
                                                    projection=self.projection, norm=self.norm)

    def forward(self, image_features):

        last_hidden_states_temp = self.prompt_learner()
        embed = last_hidden_states_temp.mean(1)
        embed_old = self.ori_embedding.mean(1)

        # Compute projection from text embedding to multi-modal projection
        text_embeds = self.projection_head_text(embed)
        text_embeds_old = self.projection_head_text(embed_old)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_embeds.t()

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        score = cos(text_embeds,text_embeds_old)
        score = 1.0-torch.mean(score)

        return logits, score
    