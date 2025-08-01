import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from models import *
import numpy as np
import attack_generator as attack
import os

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="resnet18", help="decide which network to use,choose from resnet18, resnet34")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--attack_method', type=str,default="dat", help = "choose form: dat and trades")
parser.add_argument('--model_path', default='./Res18_model/net_150.pth', help='model for white-box attack evaluation')
parser.add_argument('--method',type=str,default='dat',help='select attack setting following DAT or TRADES')

args = parser.parse_args()

transform_test = transforms.Compose([transforms.ToTensor(),])

print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
if args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

print('==> Load Model')
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "resnet34":
    model = ResNet34().cuda()
    net = "resnet34"


ckpt = torch.load(args.model_path)
model.load_state_dict(ckpt)

print(net)

model.eval()
print('==> Generate adversarial sample')


PATH_DATA='./Adv_data/cifar10/RN18'

os.makedirs(PATH_DATA, exist_ok=True)


X_adv=attack.adv_generate(model, test_loader, perturb_steps=20, epsilon=8./255, step_size=8./255 / 10,loss_fn="cent", category="Madry", rand_init=True)
np.save(os.path.join(PATH_DATA, 'Adv_cifar_PGD20_eps8.npy'), X_adv)
