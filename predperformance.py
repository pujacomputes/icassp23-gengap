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
from scipy.stats import mode
from sklearn import metrics

LOG_PREFIX="/usr/workspace/trivedi1/Fall2022/icassp23-gengap/pred_logs"
def get_probs(model, loader): 
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
            # adversarial
            if adv:
                images = adv(net, images, targets)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
    return total_loss / len(test_loader), total_correct / len(test_loader.dataset)

def test_c_domainnet(net, args):
    """Evaluate network on given corrupted dataset."""
    use_clip_mean = "clip" in args.arch.lower()
    print("=> Use Clip Mean?: ", use_clip_mean)
    corruption_accs = []
    corrs = CBAR_CORRUPTIONS if "Bar" in args.corruption_path else CORRUPTIONS
    for corruption in corrs:
        for sev in range(1, 6):
            corrupted_loader = get_corrupted_loader(
                args,
                dataset="domainnet-sketch",
                corruption_name=corruption,
                severity=sev,
                use_clip_mean=use_clip_mean,
            )

            test_loss, test_acc = test(net, corrupted_loader)
            corruption_accs.append(test_acc)
            print(
                "{},{}\tTest Loss {:.3f} | Test Error {:.3f}".format(
                    corruption, sev, test_loss, 100 - 100.0 * test_acc
                )
            )
    return np.mean(corruption_accs)

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
        choices=["cifar10", "cifar100", "domainnet-sketch", "living17"],
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
        choices=["lms", "doc", "ens", "mmd"],
        help="Specify the type of predictor",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet50", "wrn", "densenet", "resnext", "clip-RN50"],
        help="Choose architecture.",
    )
    
    parser.add_argument(
        "--save_name", default="", help="Provide an identifier for the checkpoint"
    )
    # data loader args
    parser.add_argument(
        "--eval_batch_size", default=128, type=int, help="Eval Batchsize"
    )
    parser.add_argument("--batch_size", default=128, type=int, help="Train Batchsize")
    parser.add_argument("--num_augs", default=10, type=int, help="Num Augs for LMS")
    parser.add_argument("--num_workers", default=8, type=int, help="Num Workers")
    parser.add_argument("--prefetch", action="store_true", help="Prefetch Ood Loader")
    
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

    # """
    # Create model and pretrainined checkpoint
    # """
    # net = load_model(args=args,ckpt=args.ckpt[0])
    use_clip_mean = "clip" in args.arch
    # net = torch.nn.DataParallel(net).cuda()
    # torch.backends.cudnn.benchmark = True 
    
    # print("=> Num GPUS: ", torch.cuda.device_count())
    # safety_logs_prefix = ("/usr/workspace/trivedi1/Fall2022/icassp23-gengap/pred_logs/")
    # save_name = args.save_name
    # print("=> Save Name: ", save_name) 
    
    # """
    # Get Datasets
    # """
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

    # """
    # Load the evaluation dataset
    # """
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
    # _, clean_acc = test(net=net, test_loader=clean_train_loader, adv=None)
    # _, target_acc = test(net=net, test_loader=target_loader, adv=None)
    # print("=> Clean Acc: ",clean_acc)
    # print("=> Target Acc: ",target_acc)

    if args.predictor.lower() == "lms":
        #use local manifold smoothing.
        """
        Change the target dataloader so that randaugment is applied.
        We will iterate x times over the dataset and compute their votes. 
        """
        lms_transform = get_transform(
        dataset=args.dataset, SELECTED_AUG="randaug", use_clip_mean=use_clip_mean
        )
        _, clean_test_dataset_c = get_dataloaders(
                args=args,
                train_aug="test",
                test_aug="randaug",
                train_transform=train_transform,
                test_transform=lms_transform,
                use_clip_mean=use_clip_mean,
                return_datasets=True,
        )
        target_loader = get_calibration_loader(args=args, 
            cal_dataset = args.target_dataset, 
            corruption =args.corruption, 
            severity = args.severity,
            clean_test_dataset=clean_test_dataset_c)

        """
        there is no shuffling! So we will iterate over the dataset k times
        to get the local manifold smoothness. 
        """ 
        net.eval()
        lms_preds = torch.zeros(args.num_augs,len(target_loader.dataset)) 
        target_list = []
        start_time = time.time()
        with torch.no_grad():
            for enum in range(args.num_augs):
                counter = 0
                for images, targets in target_loader:
                    images, targets = images.cuda(), targets.cuda()
                    num_samples = len(targets)
                    lms_preds[enum,counter:counter+num_samples]= net(images).data.max(1)[1].cpu() 
                    counter += num_samples 
                    if enum == 0:
                        target_list.append(targets)
                print("Collected Augs: ",enum,counter, len(target_list),np.round(time.time() - start_time,3))
            target_list = torch.cat(target_list).cpu()
        """
        Compute Agreement over Augmentations
        """
        vals, counts = mode(lms_preds,axis=0) 
        vals = vals.squeeze()
        counts = counts.squeeze()
        scores = counts / args.num_augs 
        total_correct = torch.Tensor(vals).eq(target_list).sum().item()
        acc = total_correct / target_list.shape[0]
        print("=> Target Acc: ", acc) 
        print("=> Avg Smoothness: ",scores.mean())
        
        file_name = "{prefix}/{save_name}_{seed}".format(prefix=)

    elif args.predictor.lower() == "doc":
        #use differences of confidences.
        net = load_model(args,args.ckpt[0]) 
        id_probs ,acc  = get_probs(model=net,loader=clean_test_loader)
        id_confs = id_probs.max(dim=1)[0]
        id_preds = id_probs.argmax(dim=1)
        
        ood_probs ,acc  = get_probs(model=net,loader=target_loader)
        ood_confs = ood_probs.max(dim=1)[0]

        """
        get the idx that are greater than thres
        """
        thresholds = np.arange(0, 1, 0.001) 
        acc_at_thres = []
        for thres in thresholds:
            acc_at_thres.append((id_confs >= thres).sum() / len(id_confs))
        idx = np.abs(np.array(acc_at_thres) - acc).argmin()
        found_thres = thresholds[idx]
        print("=> Optimal Thres: {0} , ID Thres Acc: {1:.3f}, True ID Acc: {2:.3f}".format(thresholds[idx],acc_at_thres[idx],acc))
        print("Target Acc: ",(ood_confs >= found_thres).sum() / len(ood_confs)) 

    elif args.predictor.lower() == "ens":
        num_models = len(args.ckpt)
        majority_criteria = num_models // 2
        print("=> Computing Ens Disagreement over {0} models!".format(num_models))
        print("=> Pls use 2 models or an odd number for now....")
        print("=> To be consider in `agreement` more than {}/{} must match!".format(majority_criteria,num_models))
        model_list = [load_model(args,ckpt) for ckpt in args.ckpt]
        """
        Generally, GDE is computed with just 2 models.
        But a trivial extension consider the ensemble in disagreement if the "majority" do not agree.
        Be careful with even number? 
        1. Load each model. 
        2. Compute and save predictions
        3. Compute the mode.
        4. Get Accuracy and Scores  
        """

        ens_preds = torch.zeros(num_models,len(target_loader.dataset)) 
        target_list = []
        start_time = time.time()
        with torch.no_grad():
            for enum in range(num_models):
                counter = 0
                total_correct = 0
                for images, targets in tqdm(target_loader,disable=False):
                    images, targets = images.cuda(), targets.cuda()
                    num_samples = len(targets)

                    pred =  model_list[enum](images).data.max(1)[1]
                    total_correct += pred.eq(targets.data).sum().item()
                    ens_preds[enum,counter:counter+num_samples]= pred.cpu() 
                    counter += num_samples 
                    if enum == 0:
                        target_list.append(targets)
                acc_tmp = total_correct / len(target_loader.dataset) 
                print("({0}) Collected Models: {1} -- Target List: {2} -- Acc: {3:.3f}".format(np.round(time.time() - start_time,3), enum,counter,acc_tmp))
            target_list = torch.cat(target_list).cpu()
        """
        Compute Agreement over Augmentations
        """
        vals, counts = mode(ens_preds,axis=0) 
        vals = vals.squeeze()
        counts = counts.squeeze()

        """
        Disagreement rate
        """
        num_disagreements = len(counts[counts <= majority_criteria]) #number of times models did not agree
        print("=> Num Disagreements! ",num_disagreements)
        ens_probs = counts / num_models #probability of top most class 
        total_correct = torch.Tensor(vals).eq(target_list).sum().item()
        acc = 1 - (num_disagreements / target_list.shape[0])
        print("=> Estimated Target Acc: ", acc) 
        print("=> Ens. Probability : ",ens_probs.mean()) 
    
    else:
        print("INVALID PREDICTOR SPECIFIED. EXITING")
        exit()
if __name__ == "__main__":
    main()