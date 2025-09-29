import numpy as np
from models import *
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchattacks import PGD, PGDL2, AutoAttack, CW, FGSM, BIM

# updated version of adv_generate, including an updated version of craft_adv
def adv_generate(
    model, 
    test_loader,
    args
    ):
    print('current attack settings:')
    print('category is: ', args.category)
    print('epsilon is: ', args.epsilon)
    print('step size is: ', args.step_size)
    print('norm is: ', args.norm)
    if args.category != 'fgsm':
        print('num_steps is: ', args.num_steps)
    model.eval()
    bool_i=0
    predicted_original_labels_list = []
    predicted_adv_labels_list = []
    target_list = []
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            # collect original target
            target_list.append(target.clone().cpu())
            # replace pgd with craft_adv
            x_adv = craft_adv(
                model = model,
                x_clean = data,
                y = target,
                args = args
            )
            # predict labels on adversarial examples (no grad needed)
            with torch.no_grad():
                logits_adv = model(x_adv)
                logits_ori = model(data)
                preds_adv = torch.argmax(logits_adv, dim=1)
                preds_ori = torch.argmax(logits_ori, dim=1)
            # collect predictions
            predicted_original_labels_list.append(preds_ori.clone().cpu())
            predicted_adv_labels_list.append(preds_adv.clone().cpu())
            if bool_i == 0:
                X_adv = x_adv.clone().cpu()
            else :
                X_adv = torch.cat((X_adv, x_adv.clone().cpu()), 0)
            bool_i +=1
    # concatenate all collected data
    all_targets = torch.cat(target_list, dim=0)
    original_labels = torch.cat(predicted_original_labels_list, dim=0)
    adv_labels = torch.cat(predicted_adv_labels_list, dim=0)
    
    # Filter: only keep samples where original prediction matches ground truth
    correct_mask = (original_labels == all_targets)
    print(f'Total samples: {len(correct_mask)}, Correctly classified: {correct_mask.sum().item()}')
    
    # Apply filter to all data
    X_adv_filtered = X_adv[correct_mask]
    original_labels_filtered = original_labels[correct_mask]
    adv_labels_filtered = adv_labels[correct_mask]
    
    return X_adv_filtered, original_labels_filtered, adv_labels_filtered

# updated version, including more types of adversarial attacks
def craft_adv(
    model,
    x_clean, 
    y, 
    args
    ):
    if args.category == 'pgd':
        attack = PGD(
            model, 
            eps=args.epsilon, 
            alpha=args.step_size, 
            steps=args.num_steps, 
            random_start=args.random_start
            )
    elif args.category == 'pgd-l2':
        attack = PGDL2(
            model, 
            eps=args.epsilon, 
            alpha=args.step_size, 
            steps=args.num_steps, 
            random_start=args.random_start
            )
    elif args.category == 'aa':
        attack = AutoAttack(
            model, 
            norm='Linf', 
            eps=args.epsilon, 
            version='rand',
            n_classes=args.num_class
            )
    elif args.category == 'aa-l2':
        attack = AutoAttack(
            model, 
            norm='L2', 
            eps=args.epsilon, 
            version='rand',
            n_classes=args.num_class
            )
    elif args.category == 'cw':
        attack = CW(
            model, 
            c=args.epsilon,
            steps=args.num_steps
            )
    elif args.category == 'fgsm':
        attack = FGSM(
            model, 
            eps=args.epsilon,
            )
    elif args.category == 'bim':
        attack = BIM(
            model, 
            eps=args.epsilon, 
            alpha=args.step_size, 
            steps=args.num_steps
            )
    x_adv = attack(x_clean, y)
    return x_adv