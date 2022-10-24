"""
Given a pretrained model, we will predict the 
generalization on a downstream dataset.

specify method, model ckpts, and seeds
save the "sample wise" confidence scores

1. Load Model
2. Load validation loader
3. Load target dataset WITH FIXED SEEDING. NO SHUFFLING!!
4. Pre-compute any required scores 
"""

"""
1. CIFAR 100-C
2. CIFAR 100-P
3. Adversarial Attacks
4. Anamoly Detection
"""
import math
import os
import random
from re import I
import time
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from safety_utils import *
from tqdm import tqdm
import torchvision.datasets as dset
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from cifar10p1 import CIFAR10p1
from clip_model import ClipModel
from utils import (
    CBAR_CORRUPTIONS_SEV,
    get_corrupted_loader,
    get_dataloaders,
    get_oodloader,
    get_transform,
    NUM_CLASSES_DICT,
    norm_dict,
    get_calibration_loader,
)
import json
from scipy.stats import mode
from sklearn.metrics import mean_squared_error, mean_absolute_error

LOG_PREFIX="/usr/workspace/trivedi1/Fall2022/icassp23-gengap/pred_logs"
def get_probs(model, loader,return_target_list=False): 
    all_feats = []
    correct = 0
    total = 0
    model.eval()
    target_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            outputs = model(x)
            all_feats.append(F.softmax(outputs, dim=1).cpu())
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
            if return_target_list:
                target_list.append(y) 
    acc = correct / total
    all_feats = torch.cat(all_feats)
    if return_target_list:
        target_list = torch.cat(target_list).cpu()
        return all_feats,acc,target_list 
    return all_feats,acc 

def get_probs_thres(model, loader): 
    all_feats = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            outputs = model(x)
            all_feats.append(F.softmax(outputs, dim=1).cpu())
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    all_feats = torch.cat(all_feats)
    return all_feats,acc 

def test(net, test_loader, adv=None):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
    return total_loss / len(test_loader), total_correct / len(test_loader.dataset)

def load_dp_ckpt(model, pretrained_ckpt):
    ckpt = torch.load(pretrained_ckpt)
    new_keys = [(k, k.replace("module.", "")) for k in list(ckpt["state_dict"].keys())]
    for old_k, new_k in new_keys:
        ckpt["state_dict"][new_k] = ckpt["state_dict"].pop(old_k)
    incompatible, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("Incompatible Keys: ", incompatible)
    print("Unexpected Keys: ", unexpected)
    return model

def arg_parser_eval():
    parser = argparse.ArgumentParser(
        description="Evaluates a Safety Metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "domainnet-sketch", "living17","cifar10ln"],
        help="Choose ID dataset",
    )
    
    parser.add_argument(
        "--target_dataset",
        type=str,
        default="id-c",
        help="Choose Target dataset",
    )
    
    parser.add_argument(
        "--predictor",
        type=str,
        default="lms",
        choices=["lms", "doc", "ens", "mmd",'voting','gde','gde-cls'],
        help="Specify the type of predictor",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet50", "wrn", "densenet", "resnext", "clip-RN50","resnet18"],
        help="Choose architecture.",
    )
    
    parser.add_argument(
        "--save_name", default="", help="Provide an identifier for the checkpoint"
    )
    # data loader args
    parser.add_argument(
        "--eval_batch_size", default=128, type=int, help="Eval Batchsize"
    )
    parser.add_argument("--learning_rate", default=1.0, type=float, help="Label Noise GDECLS")
    parser.add_argument("--decay", default=0.0001, type=float, help="Label Noise GDECLS")
    parser.add_argument("--momentum", default=0.9, type=float, help="Label Noise GDECLS")
    parser.add_argument("--batch_size", default=128, type=int, help="Train Batchsize")
    parser.add_argument("--epochs", default=200, type=int, help="CLS Epochs")
    parser.add_argument("--num_augs", default=10, type=int, help="Num Augs for LMS")
    parser.add_argument("--num_workers", default=8, type=int, help="Num Workers")
    parser.add_argument("--prefetch", action="store_true", help="Prefetch Ood Loader")
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--no_bias", dest="use_bias", action="store_false") 
    parser.add_argument(
        "--cal_datasets",
        nargs="*",
        help="Datasets for Regressor",
        required=False,
    )
    parser.add_argument(
        "--ckpt",
        nargs="*",
        help="Checkpoint path for resume / test.",
    )
    parser.add_argument("--corruption", default="gaussian_blur", type=str, help="corruption name, Used for Getting CIFAR10-C")
    parser.add_argument("--severity", default=5, type=int, help="severity, Used for Getting CIFAR10-C")
    parser.add_argument("--base_idx", default=0, type=int, help="Hold out model for GDE")
    parser.add_argument("--num_cls", default=10, type=int, help="Num Noisy CLS for Soup")
    parser.add_argument("--label_noise", default=0.02, type=float, help="Label Noise GDECLS")
    parser.add_argument("--cls_label_noise", default=0.02, type=float, help="Label Noise GDECLS")

    args = parser.parse_args()
    return args

def load_model(args,ckpt):
    if args.arch.lower() == "resnet50":
        net = timm.create_model(args.arch, pretrained=False)
        net.reset_classifier(NUM_CLASSES_DICT[args.dataset])
        net = load_dp_ckpt(net, pretrained_ckpt=ckpt).cuda()  # do this so there is not a conflict.
    elif args.arch.lower() == "resnet18":
        net = timm.create_model(args.arch, pretrained=False)
        net.reset_classifier(NUM_CLASSES_DICT[args.dataset])
        net = load_dp_ckpt(net, pretrained_ckpt=ckpt).cuda()  # do this so there is not a conflict.
    if "clip" in args.arch:
        encoder_type = args.arch.split("-")[-1]
        print("\t => Clip Encoder: ", encoder_type)
        print("\t => Using Default Clip Ckpt!!")
        net = ClipModel(model_name=encoder_type, scratch=False)
        net.reset_classifier(NUM_CLASSES_DICT[args.dataset])
        net = load_dp_ckpt(net, pretrained_ckpt=ckpt).cuda()
    net.eval()
    return net

def perform_voting(model_list, loader):
    """
    Get predictions
    """
    with torch.no_grad():
        predictions = [[] for  _ in model_list]
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            for enum, m in enumerate(model_list):
                pred = m(x).argmax(1)
                predictions[enum].append(pred)
        for enum, p in enumerate(predictions):
            predictions[enum] = torch.cat(p)
        predictions = torch.stack(predictions).cpu().numpy()  # num models x n

    """
    Vote on predictions
    """
    pred_acc_mtrx = np.zeros(len(model_list))
    true_acc_mtrx = np.zeros(len(model_list))
    gde_scores = [[] for _ in range(len(model_list))]
    for s_num,_ in enumerate(model_list):
        #s_num is the held-out model
        #sub_idx contains the model idx used for voting
        sub_idx = [i for i,_ in enumerate(model_list) if i != s_num]
        votes = mode(predictions[sub_idx, :], axis=0)[0].reshape(-1)
        base_preds = predictions[s_num]
        disagreements = base_preds == votes #disagreement over n-1 
        gde_scores[s_num] = disagreements
        predicted_acc = (disagreements.sum()) / disagreements.shape[0]
        _, true_acc = test(model_list[s_num], loader)
        pred_acc_mtrx[s_num] = predicted_acc
        true_acc_mtrx[s_num] = true_acc
    true_acc_mtrx = true_acc_mtrx.reshape(-1,1)
    pred_acc_mtrx = pred_acc_mtrx.reshape(-1,1)
    MSE = mean_squared_error(y_true=true_acc_mtrx,y_pred=pred_acc_mtrx)
    MAE = mean_absolute_error(y_true=true_acc_mtrx,y_pred=pred_acc_mtrx)
    avg_pred_acc = pred_acc_mtrx.mean()
    avg_true_acc = true_acc_mtrx.mean()
    print("=> Avg Pred Acc: ",avg_pred_acc)
    print("=> Avg True Acc: ",avg_true_acc)

    """
    We will average the round-robin disagreement 
    scores to get a better score for each sample.
    """
    gde_scores = np.stack(gde_scores) #num models x num_samples
    return gde_scores, MSE, MAE, pred_acc_mtrx, true_acc_mtrx

def perform_voting_one_vs_one_clfs(net, clf_list, loader,base_clf_idx=0):
    """
    Get predictions.
    We treat one model as the pretrained model.
    All other models are considered the "check" members of the ensemble.
    We will compute the GDE score with each check member separately.
    We then average the disagreement scores to compute a threshold.
    e.g., what is the expected GDE score over the ensemble.

    base_model_idx tells us which model will be held out of the set of check models 
    """
    with torch.no_grad():
        predictions = [[] for  _ in clf_list]
        total_correct = 0
        total_samples = 0
        target_list = []  
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            pen = torch.nn.functional.adaptive_avg_pool2d(
                net.forward_features(x), 1
            )
            pen = torch.squeeze(pen)
            for enum, clf in enumerate(clf_list):
                pred = clf(pen).argmax(1)
                predictions[enum].append(pred)
                if enum == base_clf_idx:
                    total_correct += pred.eq(y.data).sum().item()
                    total_samples += len(y)
                    target_list.append(y)
        for enum, p in enumerate(predictions):
            predictions[enum] = torch.cat(p)
    target_list = torch.cat(target_list).cpu()
    acc = total_correct / total_samples 
    """
    Compute disagreement with all the other models
    """
    base_preds= predictions[base_clf_idx] 

    gde_disagreements = []
    for enum in range(len(clf_list)): 
        if enum !=base_clf_idx: 
            check_preds = predictions[enum]
            gde_disagreements.append(base_preds.eq(check_preds)) 
    gde_disagreements = torch.stack(gde_disagreements).float().cpu()  # (num models-1) x n {0,1} binary array of GDE predictions
    """
    We will average the round-robin disagreement 
    scores to get a better score for each sample.
    """
    return gde_disagreements, predictions[base_clf_idx].cpu(),target_list, acc

def perform_voting_one_vs_one(model_list, loader,base_model_idx=0):
    """
    Get predictions.
    We treat one model as the pretrained model.
    All other models are considered the "check" members of the ensemble.
    We will compute the GDE score with each check member separately.
    We then average the disagreement scores to compute a threshold.
    e.g., what is the expected GDE score over the ensemble.

    base_model_idx tells us which model will be held out of the set of check models 
    """
    with torch.no_grad():
        predictions = [[] for  _ in model_list]
        total_correct = 0
        total_samples = 0
        target_list = []  
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            for enum, m in enumerate(model_list):
                pred = m(x).argmax(1)
                predictions[enum].append(pred)
                if enum == base_model_idx:
                    total_correct += pred.eq(y.data).sum().item()
                    total_samples += len(y)
                    target_list.append(y)
        for enum, p in enumerate(predictions):
            predictions[enum] = torch.cat(p)
    target_list = torch.cat(target_list).cpu()
    acc = total_correct / total_samples 
    """
    Compute disagreement with all the other models
    """
    base_preds= predictions[base_model_idx] 

    gde_disagreements = []
    for enum in range(len(model_list)): 
        if enum != base_model_idx: 
            check_preds = predictions[enum]
            gde_disagreements.append(base_preds.eq(check_preds)) 
    gde_disagreements = torch.stack(gde_disagreements).float().cpu()  # (num models-1) x n {0,1} binary array of GDE predictions
    """
    We will average the round-robin disagreement 
    scores to get a better score for each sample.
    """
    return gde_disagreements, predictions[base_model_idx].cpu(),target_list, acc

def get_ens_acc(model_list, loader):
    """
    Get predictions.
    We treat one model as the pretrained model.
    All other models are considered the "check" members of the ensemble.
    We will compute the GDE score with each check member separately.
    We then average the disagreement scores to compute a threshold.
    e.g., what is the expected GDE score over the ensemble.

    base_model_idx tells us which model will be held out of the set of check models 
    """
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            pred = model_list[0](x)
            for m in model_list[1:]:
                pred += F.softmax(m(x),dim=1)
            pred /= len(model_list)
            pred = pred.argmax(1)
            total_correct += pred.eq(y.data).sum().item()
            total_samples += len(y)
    acc = total_correct / total_samples 
    return acc

def get_lms(net,randaug_loader,args):
    """
    We expect a dataloader that uses randaugment to get lms scores.
    There is no shuffling! So we will iterate over the dataset k times
    to get the local manifold smoothness. 
    """ 
    net.eval()
    lms_preds = torch.zeros(args.num_augs,len(randaug_loader.dataset)) 
    target_list = []
    start_time = time.time()
    print()
    print("=> Begin LMS.")
    with torch.no_grad():
        for enum in range(args.num_augs):
            counter = 0
            for images, targets in randaug_loader:
                images, targets = images.cuda(), targets.cuda()
                num_samples = len(targets)
                lms_preds[enum,counter:counter+num_samples]= net(images).data.max(1)[1].cpu() 
                counter += num_samples 
                if enum == 0:
                    target_list.append(targets)
            print("({0}) Collected Augs: {1}".format(np.round(time.time() - start_time,3),enum))
        target_list = torch.cat(target_list).cpu()
    print()
    return lms_preds, target_list 

def extract_features(args, model, loader, train_aug, train_transform):
    if train_aug in ["cutmix", "mixup", "cutout"]:
        transform = train_transform
    else:
        transform = None
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(loader, disable=True):
            data = data.cuda()
            if transform:
                if train_aug in ["cutmix", "mixup"]:
                    images, targets = transform(images, target=targets)
                if train_aug == "cutout":
                    images = transform(images)
            if args.arch == "resnet50":
                # using a timm model. called the
                reps = torch.nn.functional.adaptive_avg_pool2d(
                    model.module.forward_features(data), 1
                )
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
            if args.arch == "resnet18":
                # using a timm model. called the
                reps = torch.nn.functional.adaptive_avg_pool2d(
                    model.forward_features(data), 1
                )
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
            if "clip" in args.arch:
                # using a timm model. called the
                reps = model.module.get_features(data)
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
    # features = torch.nn.functional.adaptive_avg_pool2d(np.squeeze(np.concatenate(features)),1)
    features = np.squeeze(np.concatenate(features))
    labels = np.concatenate(labels)
    return features, labels
def create_sparse_clfs(classifier_pool):
    num_dims = classifier_pool[0].weight.data.shape[1]
    sparsity_feats = num_dims // len(classifier_pool) 
    num_classes = classifier_pool[0].weight.data.shape[0]
    for enum, cls in enumerate(classifier_pool):
        """
        if we are at the end of the classifiers, 
        then we will just let the classifier use a few extra features.
        """

        start_idx = enum * sparsity_feats

        if enum == len(classifier_pool)-1:
            end_idx = cls.weight.data.shape[1]

        else:
            end_idx = (enum+1) * sparsity_feats

        """
        Now, set all weights to 0, and then replace specified indices, with these values. 
        """

        with torch.no_grad():
            empty = torch.zeros(num_classes,end_idx -start_idx)
            torch.nn.init.kaiming_uniform_(empty, a=math.sqrt(5))
            torch.nn.init.constant_(cls.weight.data,0.0)
            cls.weight[:,start_idx:end_idx] = empty
            assert empty.norm(), cls.weight.norm()
            # print(enum, start_idx, end_idx,np.round(cls.weight.norm().cpu().numpy(),4))
    return classifier_pool


def linear_probe_soup(args, net, train_loader,test_loader, train_aug, train_transform):
    net.eval()
    print("=> Extracting Features...")
    train_features, train_labels = extract_features(args,model=net, loader=train_loader,train_aug=train_aug,train_transform=train_transform)
    test_features, test_labels = extract_features(args,model=net, loader=test_loader,train_aug='test',train_transform=None)
    print(train_features.shape)

    rep_test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_features),torch.Tensor(test_labels).long())
    rep_test_dataloader = torch.utils.data.DataLoader(rep_test_dataset,batch_size=args.batch_size,shuffle=True) 

    """
    Training each classifer separately on a different noise instance of the data. 
    """
    classifier_pool = [torch.nn.Linear(train_features.shape[1],NUM_CLASSES_DICT[args.dataset],bias=args.use_bias).to(DEVICE) for _ in range(args.num_cls)]
    with torch.no_grad():
        print("Pre Sparse Classifier Inits: ",[np.round(c.weight.data.norm().data.item(),4) for c in classifier_pool])
    classifier_pool = create_sparse_clfs(classifier_pool=classifier_pool)
    with torch.no_grad():
        print("Sparsified: ",[np.round(c.weight.data.norm().data.item(),4) for c in classifier_pool])
    
    with torch.no_grad():
        norms = [c.weight.norm() for c in classifier_pool]
    print("=> Untrained Clf Norms: ",norms)
    num_relabel = int(len(train_labels) * args.cls_label_noise)
    print("=> Num Relabeled: ",num_relabel)
    for enum,clf in enumerate(classifier_pool):
        print("Training Clf: , ", enum)
        """
        Relabel a small portion of the training dataset.
        """
        relabel_idx = np.random.choice(a=np.arange(len(train_labels)),size=num_relabel,replace=False) 
        relabel_labels = np.random.choice(a=np.arange(10),size=num_relabel,replace=True)
        noisy_targets = train_labels.copy()
        noisy_targets[relabel_idx] = relabel_labels
        rep_train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_features),torch.Tensor(noisy_targets).long())
        rep_train_dataloader = torch.utils.data.DataLoader(rep_train_dataset,batch_size=args.batch_size,shuffle=True)  

        """
        Set-up training
        """ 
        optimizer = torch.optim.SGD(
            clf.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.decay,
            nesterov=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = args.epochs,
        )
        criterion = torch.nn.CrossEntropyLoss()
        for epochs in range(args.epochs):
            loss_avg = 0
            for batch_idx, (data, target) in enumerate(rep_train_dataloader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                loss = 0
                loss =  criterion(clf(data),target) 
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_avg += loss 
            train_loss,train_acc = test(clf, rep_train_dataloader) 
            _,test_acc = test(clf, rep_test_dataloader) 
            print("Clf: {0} -- Train Acc: {1:.3f} -- Test Acc:{2:.3f} -- Train Loss:{3:.3f}".format(enum,train_acc,test_acc,train_loss))

        print("="*30)
        # net.fc = clf
        # train_loss,train_acc = test(net, train_loader) 
        # _,test_acc = test(net, test_loader) 
        print("Finished Training Clf: {0}, Final Train: {1:.3f}, Final Test: {2:.3f}".format(enum, train_acc, test_acc))
        print("="*30)
        del rep_train_dataset,rep_train_dataloader,optimizer, scheduler 
    with torch.no_grad():
        norms = [c.weight.norm().data for c in classifier_pool]
    print("=> Trained Clf Norms: ",norms)

    """
    After all individual clf are trained.
    We are going to return them, so that we may estimate error with them. 
    """

    return classifier_pool 
def main():
    """
    Load the saved checkpoint.
    Load clean CIFAR.
    Provide options to perform the other ML safety evaluations.
    We will need to use their checkpoints.
    """
    args = arg_parser_eval()
    for arg in sorted(vars(args)):
        print("=> " ,arg, getattr(args, arg))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    net = load_model(args,args.ckpt[0])
    use_clip_mean = "clip" in args.arch
    
    """
    Get Datasets
    """
    train_transform = get_transform(
        dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean
    )
    test_transform = get_transform(
        dataset=args.dataset, SELECTED_AUG="test", use_clip_mean=use_clip_mean
    )
    clean_train_loader, clean_test_loader = get_dataloaders(
        args=args,
        train_aug="test",
        test_aug="test",
        train_transform=train_transform,
        test_transform=test_transform,
        use_clip_mean=use_clip_mean,
    )

    """
    Load the evaluation dataset
    """
    _, clean_test_dataset_c = get_dataloaders(
                args=args,
                train_aug="test",
                test_aug="test",
                train_transform=train_transform,
                test_transform=test_transform,
                use_clip_mean=use_clip_mean,
                return_datasets=True,
    )
    target_loader = get_calibration_loader(args=args, 
        cal_dataset = args.target_dataset, 
        corruption =args.corruption, 
        severity = args.severity,
        clean_test_dataset=clean_test_dataset_c)
    print("=> DONE LOADING ALL DATASETS") 
    # _, id_acc = test(net=net, test_loader=clean_test_loader, adv=None)
    # _, target_acc = test(net=net, test_loader=target_loader, adv=None)

   
    if args.predictor.lower() == "gde-cls":
        if args.num_cls < 2:
            print("ONLY 1 MODEL PASSED. NOT AN ENSEMBLE. EXITING")
            exit()
        print("=> NUM CLS: ",args.num_cls)
        print("=> Base cls idx: ",args.base_idx) #irrelevant

        net = load_model(args,args.ckpt[0])
        clf_soup = linear_probe_soup(args, net, clean_train_loader, clean_test_loader, "test", test_transform) 
        id_disagreements, id_base_preds, id_labels, id_acc =perform_voting_one_vs_one_clfs(net, 
            clf_list=clf_soup, 
            loader=clean_test_loader,
            base_clf_idx=0)
        id_scores = torch.mean(id_disagreements,dim=0) 
        id_scores = np.round(id_scores,4) 
        
        target_disagreements, target_base_preds, target_labels, target_acc =perform_voting_one_vs_one_clfs(net, 
            clf_list=clf_soup, 
            loader=target_loader,
            base_clf_idx=0)
        target_scores = torch.mean(target_disagreements,dim=0) 
        target_scores = np.round(target_scores,4) 
  
        """
        Compute the threshold
        """
        print('=> Computing Threshold')
        thresholds = np.arange(0, 1, 1/(args.num_cls-1)) 
        acc_at_thres = []
        for thres in thresholds:
            acc_at_thres.append((id_scores >= thres).sum() / len(id_scores))
        idx = np.abs(np.array(acc_at_thres) - id_acc).argmin()
        found_thres = np.round(thresholds[idx],4)
        print("=> Optimal Thres: {0} , ID Thres Acc: {1:.3f}, True ID Acc: {2:.3f}".format(found_thres,acc_at_thres[idx],id_acc))
        
        """
        Get the predicted TARGET DIST ACC using the threshold.
        """
        num_disagreements = len(target_scores[target_scores >= found_thres]) #number of times models did not agree
        print("=> Num Disagreements (Threshold)! ",num_disagreements)
        target_ground_truth = target_base_preds.eq(target_labels)
        pred_target_acc = (num_disagreements / target_labels.shape[0])
        print("True Target Acc: {0:.3f} -- Pred Thres Acc: {1:.3f}".format(target_acc,pred_target_acc))

        """
        Save Score and relevant info to txt.
        """ 
        consolidated_log_path ="/usr/workspace/trivedi1/Fall2022/icassp23-gengap/pred_logs/consolidated.csv"
        save_name = "{method}-{base_idx}-{num_models}_{label_noise}_{save_name}_{corruption}_{severity}_{seed}".format(save_name=args.save_name,
            base_idx=args.base_idx,
            corruption=args.corruption,
            severity=args.severity, 
            seed=args.seed,
            method=args.predictor,
            label_noise=args.cls_label_noise,
            num_models=args.num_cls)
        print("=>Save Name: ",save_name)
        # print("=> ENS ID Acc: ",np.round(ens_acc_id,4)) 
        # print("=> ENS Target Acc: ",np.round(ens_acc_target,4))
        mae_diff = np.abs(target_acc - pred_target_acc)
        save_str = "{save_name},{true_target_acc:.4f},{pred_target_acc:.4f},{mae:.4f}\n".format(save_name=save_name,true_target_acc=target_acc, pred_target_acc=pred_target_acc,mae=mae_diff)
        with open(consolidated_log_path, "a") as f:
            f.write(save_str)

        file_name = "{prefix}/{method}_{save_name}_{corruption}_{severity}_{seed}.txt".format(prefix=LOG_PREFIX,
            save_name=args.save_name,
            corruption=args.corruption,
            severity=args.severity, 
            seed=args.seed,
            method=args.predictor)
        #scores, predicted as correct, ground_truth_is_correct_pred, predicted_class, ground_truth_class
        #arr = np.column_stack((target_scores, target_scores >= found_thres, target_ground_truth,target_vals, target_list))
        arr = np.column_stack((target_scores, target_scores >= found_thres, target_ground_truth,target_base_preds, target_labels))
        print("Saving Clfs Ens Scores!")
        print("File Name: {0}".format(file_name))
        print("Arr Size: ",arr.shape)

        header_dict ={
            "thres":found_thres,
            "pred_target_acc":np.round(pred_target_acc,4),
            "true_target_acc":np.round(target_acc,4),
            "target_dataset":args.target_dataset,
            "corruption":args.corruption,
            "severity":args.severity,
            # "ens_target_acc":ens_acc_target,
            # "ens_id_acc":ens_acc_id
        }
        header_str = json.dumps(header_dict)
        np.savetxt(file_name,X=arr,delimiter=",",fmt="%.4f",header=header_str)  
    # else:
    #     print("INVALID PREDICTOR SPECIFIED. EXITING")
    #     exit()
if __name__ == "__main__":
    main()