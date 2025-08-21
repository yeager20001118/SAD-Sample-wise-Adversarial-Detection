import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from models import *
import numpy as np
import attack_generator_jiacheng as attack
import os

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="resnet18", help="decide which network to use,choose from resnet18, resnet34")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--model_path', default='./Res18_ckpt/net_150.pth', help='model for white-box attack evaluation')

# Modification: adding configurations of adversarial attacks
parser.add_argument('--epsilon', default=8, type=int, help='perturbation')
parser.add_argument('--num-steps', default=20, type=int, help='perturb number of steps')
parser.add_argument('--num-class', default=10, type=int, help='number of classes')
parser.add_argument('--step-size', default=1, type=int, help='perturb step size')
parser.add_argument('--category', type=str, default='pgd', help='select attack category')
parser.add_argument("--random-start", action="store_true")
parser.add_argument('--norm', type=str, default='linf', help='select attack norm')
args = parser.parse_args()

def main():
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
        net = "Res18"
    if args.net == "resnet34":
        model = ResNet34().cuda()
        net = "Res34"

    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt)

    print(net)

    model.eval()
    print('==> Generate adversarial sample')

    PATH_DATA='./Adv_data/{}/{}/with_labels'.format(args.dataset, net)
    ATTACK_FILENAME = 'Adv_{}_{}_{}_eps{}_{}.npz'.format(args.dataset, args.category, args.num_steps, args.epsilon, args.norm)

    os.makedirs(PATH_DATA, exist_ok=True)

    args.epsilon = args.epsilon / 255
    args.step_size = args.epsilon / 5

    X_adv, original_labels, predicted_labels = attack.adv_generate(
        model = model, 
        test_loader = test_loader,
        args = args
    )

    np.savez(
        os.path.join(PATH_DATA, ATTACK_FILENAME),
        X_adv=X_adv.detach().cpu().numpy(),
        original_labels=original_labels.detach().cpu().numpy(),
        predicted_labels=predicted_labels.detach().cpu().numpy(),
    )
    print('Adversarial examples (with labels) saved to: {}'.format(os.path.join(PATH_DATA, ATTACK_FILENAME)))

if __name__ == "__main__":
    main()
