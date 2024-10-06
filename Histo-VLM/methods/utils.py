import numpy as np
import torch
from torch import Tensor as tensor
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, f1_score, recall_score

from tqdm import tqdm

def specificity(refs, preds):
    cm = confusion_matrix(refs, preds )
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    return specificity

def average_task_results(list_results):
    metrics_name = list(list_results[0].keys())

    out = {}
    for iMetric in metrics_name:
        values = np.concatenate([np.expand_dims(np.array(iFold[iMetric]), -1) for iFold in list_results], -1)
        out[(iMetric + "_avg")] = np.round(np.mean(values, -1), 4).tolist()
        out[(iMetric + "_std")] = np.round(np.std(values, -1), 4).tolist()

    print('Metrics: aca=%2.4f(%2.4f) - kappa=%2.3f(%2.3f) - macro f1=%2.3f(%2.3f)' % (
        out["aca_avg"], out["aca_std"], out["kappa_avg"], out["kappa_std"], out["f1_avg_avg"], out["f1_avg_std"]))

    return out

def classification_metrics(refs, preds, method="others"):

    # Kappa quadatic
    if method == "lp":
        k = np.round(cohen_kappa_score(refs, preds, weights="quadratic"), 3)
    else:
        k = np.round(cohen_kappa_score(refs, np.argmax(preds, -1), weights="quadratic"), 3)

    # Confusion matrix
    if method == "lp":
        cm = confusion_matrix(refs, preds)
    else:
        cm = confusion_matrix(refs, np.argmax(preds, -1))
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    # Accuracy per class - and average
    acc_class = list(np.round(np.diag(cm_norm), 3))
    aca = np.round(np.mean(np.diag(cm_norm)), 4)

    if method == "lp":
        # recall
        recall_class = [np.round(recall_score(refs == i, preds== i), 3) for i in np.unique(refs)]
        # specificity
        specificity_class = [np.round(specificity(refs == i, preds == i), 3) for i in np.unique(refs)]
    else:
        # recall
        recall_class = [np.round(recall_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]
        # specificity
        specificity_class = [np.round(specificity(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]

    # class-wise metrics
    if method == "lp":
        auc_class = [np.round(roc_auc_score(refs == i, preds == i), 3) for i in np.unique(refs)]
        f1_class = [np.round(f1_score(refs == i, preds== i), 3) for i in np.unique(refs)]
    else:
        auc_class = [np.round(roc_auc_score(refs == i, preds[:, i]), 3) for i in np.unique(refs)]
        f1_class = [np.round(f1_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]

    metrics = {"aca": aca, "kappa": k, "acc_class": acc_class, "f1_avg": np.mean(f1_class),
               "auc_avg": np.mean(auc_class),
               "auc_class": auc_class, "f1_class": f1_class,
               "sensitivity_class": recall_class, "sensitivity_avg": np.mean(recall_class),
               "specificity_class": specificity_class, "specificity_avg": np.mean(specificity_class),
               "cm": cm, "cm_norm": cm_norm}
    

    return metrics

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def search_hp_tip(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def get_one_hot(y_s: torch.tensor, num_classes: int):
    """
        args:
            y_s : torch.Tensor of shape [n_task, shot]
        returns
            y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device, dtype=torch.float16)
    print(one_hot)

    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
    return one_hot

def compute_centroids_alpha(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0))
    centroids = (one_hot*z_s/ one_hot.sum(-2, keepdim=True)).sum(1)  # [batch, K, d]
    return centroids


def compute_centroids(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).transpose(1, 2).to(z_s.dtype) 
    # centroids = one_hot.bmm(z_s) / one_hot.sum(-1, keepdim=True)  # [batch, K, d]
    centroids = one_hot.bmm(z_s)  # [batch, K, d]
    return centroids


