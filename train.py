from asyncio import protocols
import os
import time
import torch
import timm
import tqdm 
import argparse
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from utils import *
import pdb 
from clip_model import ClipModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Wrapped into function so we 
can call it for fine-tuning too.
"""
PREFIX="/usr/workspace/trivedi1/Fall2022/icassp23-gengap/"
def train_loop(args,protocol,save_name,log_path, net, optimizer,scheduler,start_epoch,end_epoch,train_loader, test_loader, train_aug, train_transform):

    best_acc = 0
    print('=> Beginning training from epoch:', start_epoch + 1)
    if train_aug in ['cutmix','mixup','cutout']:
        transform = train_transform
    else:
        transform = None
    if train_aug in ['cutmix','mixup']:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(start_epoch, end_epoch):
        begin_time = time.time() 
        net.train()
        loss_ema = 0.
        for _, (images, targets) in tqdm.tqdm(enumerate(train_loader),disable=True):
            optimizer.zero_grad()
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            #use cutmix or mixup
            if transform:
                if train_aug in ['cutmix','mixup']:
                    images, targets= transform(images,target=targets)
                if train_aug == 'cutout':
                    images = transform(images)
            logits = net(images)
            loss = criterion(logits, targets) 
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1

        test_loss, test_acc = test(net, test_loader) 
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_time,
                loss_ema,
                test_loss,
                100 - 100. * test_acc,
            ))
      
        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f} '
            .format((epoch + 1), int(time.time() - begin_time), loss_ema,
                    test_loss, 100 - 100. * test_acc))

    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.arch,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'protocol':args.protocol
    }
    return net,checkpoint                              

def extract_features(args, model,loader,train_aug,train_transform):
    if train_aug in ['cutmix','mixup','cutout']:
        transform = train_transform
    else:
        transform = None
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm.tqdm(loader,disable=True):
            data = data.cuda()
            if transform:
                if train_aug in ['cutmix','mixup']:
                    images, targets= transform(images,target=targets)
                if train_aug == 'cutout':
                    images = transform(images)
            if args.arch == 'resnet50':
                #using a timm model. called the 
                reps = torch.nn.functional.adaptive_avg_pool2d(model.module.forward_features(data),1)
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
            if "clip" in args.arch:
                #using a timm model. called the
                reps = model.module.get_features(data)
                features.append(reps.detach().cpu().numpy())
                labels.append(targets.detach().cpu().numpy())
    # features = torch.nn.functional.adaptive_avg_pool2d(np.squeeze(np.concatenate(features)),1)
    features = np.squeeze(np.concatenate(features))
    labels = np.concatenate(labels)
    return features, labels


def main():
    args = arg_parser()
    for arg in sorted(vars(args)):
        print("=> " ,arg, getattr(args, arg))
    print("=> Device: ",DEVICE)
    print("=> Num GPUS: ",torch.cuda.device_count())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)


    if args.pretrained_ckpt.lower() == "scratch":
        if args.arch.lower()  == 'resnet50':
            print("\t*** Using UNTRAINED RESNET50 Model!!")
            net = timm.create_model(args.arch,pretrained=False)
        if args.arch.lower()  == 'resnet18':
            print("=> *** Using UNTRAINED Resnet18 Model!! ***")
            net = timm.create_model(args.arch,pretrained=False)
        if 'clip' in args.arch:
            encoder_type = args.arch.split("-")[-1]
            print("\t => Clip Encoder: ",encoder_type)
            print("\t => Using Scratch Clip Encoder!") 
            net = ClipModel(model_name=encoder_type,scratch=True)
    else:
        if args.arch.lower() == 'resnet50':
            net = timm.create_model(args.arch,pretrained=False)
            net = load_moco_ckpt(model=net, args=args)
            print("\t*** Using MoCoV2 RN50 Pretrained Model!!")
        if 'clip' in args.arch:
            encoder_type = args.arch.split("-")[-1]
            print("\t => Clip Encoder: ",encoder_type)
            print("\t => Using Default Clip Ckpt!!") 
            net = ClipModel(model_name=encoder_type,scratch=False)
    use_clip_mean = "clip" in args.arch 
    lp_aug_name = args.train_aug
    if "ln" in args.dataset:
        dataset_name = "{}-{}".format(args.dataset, args.label_noise)
    else:
        dataset_name = args.dataset 
    save_name =  dataset_name \
        + '_' + args.arch \
        + "_" + args.ft_train_aug \
        + "_" + str(args.ft_epochs) \
        + "_" + str(args.ft_learning_rate) \
        + "_" + str(args.ft_decay) \
        + "_" + str(args.seed) 
    
    print("******************************")
    print(save_name)
    print("******************************")
 
    """
    Throw away classifier.
    Create new classifier with number of classes.
    """
    net.reset_classifier(NUM_CLASSES_DICT[args.dataset])
    print('Reset Classifer: ',net.get_classifier()) 
    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
    torch.backends.cudnn.benchmark = True
   
    """
    Performing Fine-tuing Training!
    """
 
    log_path = os.path.join("{}/logs".format(PREFIX),
        "ft+" + save_name + '_training_log.csv') 
    """
    Select FT Augmentation Scheme.
    """
    ft_train_transform = get_transform(dataset=args.dataset, 
        SELECTED_AUG=args.ft_train_aug,
        use_clip_mean=use_clip_mean)

    ft_test_transform = get_transform(dataset=args.dataset, 
        SELECTED_AUG=args.ft_test_aug,
        use_clip_mean=use_clip_mean)
            
    ft_train_loader, ft_test_loader = get_dataloaders(args=args, 
        train_aug=args.ft_train_aug,
        test_aug=args.ft_test_aug, 
        train_transform=ft_train_transform,
        test_transform=ft_test_transform,
        use_clip_mean=use_clip_mean) 

    _, test_acc = test(net, ft_test_loader)
    print("=> Epoch 0 Test Acc: ",test_acc)
    
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.ft_learning_rate,
        momentum=args.ft_momentum,
        weight_decay=args.ft_decay,
        nesterov=True)
        
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max = args.ft_epochs,
    # )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
    start_epoch = 0

    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    """
    Perform Fine-Tuning Training 
    """ 
    net, ckpt = train_loop(args =args,
        protocol='ft',
        save_name = "ft+"+save_name,
        log_path=log_path,
        net = net, 
        optimizer = optimizer,
        scheduler = scheduler,
        start_epoch = start_epoch,
        end_epoch = args.ft_epochs,
        train_loader = ft_train_loader, 
        test_loader = ft_test_loader, 
        train_aug = args.ft_train_aug, 
        train_transform=ft_train_transform)

    """
    Save FT Final Ckpt.
    """
    s = "ft+{save_name}_final_checkpoint_{epoch:03d}_pth.tar".format(save_name=save_name,epoch=args.ft_epochs)
    save_path = os.path.join(args.save, s)
    torch.save(ckpt, save_path) 
    ft_train_loss, ft_train_acc = test(net, ft_train_loader)
    ft_test_loss, ft_test_acc = test(net, ft_test_loader)
    
    """
    Perform ID + OOD Evaluation!
    """ 
    with open("logs/consolidated.csv","a") as f:
        write_str = [save_name.replace("_",","),
            args.eval_dataset,
            ft_train_acc,
            ft_test_acc,
            ft_train_loss,
            ft_test_loss,
            ]
        write_str = [str(i) for i in write_str]
        write_str = ",".join(write_str)
        f.write("{}\n".format(write_str))
        print(write_str)

if __name__ == '__main__':
    main()