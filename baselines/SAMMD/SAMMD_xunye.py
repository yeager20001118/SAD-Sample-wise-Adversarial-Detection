import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from utils_new import *
from models import *

np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate for C2STs")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n", type=int, default=10, help="number of samples in one set")
opt = parser.parse_args()
print(opt)
dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False
N_per = 100 # permutation times
alpha = 0.05 # test threshold
N1 = opt.n # number of samples in one set
K = 20 # number of trails
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)


ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
Results = np.zeros([K])

# Extract semantic features from trained model
model = semantic_ResNet18().cuda()
ckpt = torch.load('./adv/Res18_model/net_150.pth') # download targeted model
model.load_state_dict(ckpt)
model.eval()    

transform_test = transforms.Compose([transforms.ToTensor(),])
testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True, num_workers=0) # shuffle P

for i, (imgs, labels) in enumerate(test_loader):
    data_org = imgs
    data_lbl = labels

P = data_org.cuda()
n_P = P.shape[0]

batch_size = 128
n_batches = (n_P - 1) // batch_size + 1
outputs = []

with torch.no_grad():
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_P)
        batch = P[start_idx:end_idx]
        batch_output = model(batch)
        outputs.append(batch_output)

P_features = torch.cat(outputs, dim=0) # dim: (10000, 10)

# adv attack data
Q = np.load("./adv/Adv_data/cifar10/RN18/Adv_cifar_PGD20_eps8.npy")
Q = torch.from_numpy(Q).float().cuda()
n_Q = Q.shape[0]

n_batches = (n_Q - 1) // batch_size + 1
outputs = []

with torch.no_grad():
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_Q)
        batch = Q[start_idx:end_idx]
        batch_output = model(batch)
        outputs.append(batch_output)

Q_features = torch.cat(outputs, dim=0) # dim: (10000, 10)

# uncomment if type I error 
# P = Q
# P_features = Q_features

np.random.seed(10086)
ind_Q = np.random.choice(len(Q_features), len(Q_features), replace=False) # shuffle Q
Q_features = Q_features[ind_Q]
Q = Q[ind_Q]

adversarial_loss = torch.nn.CrossEntropyLoss()
print(P_features.shape)
print(Q_features.shape)
for kk in range(K):
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=1102 * (kk + 10) + N1)

    if cuda:
        adversarial_loss.cuda()

    # samples N1 from P
    tr_IND = np.random.choice(len(P_features), N1, replace=False)
    te_IND = np.delete(np.arange(len(P_features)), tr_IND)
    P_tr = P[tr_IND].view(N1, -1)
    P_te = P[te_IND].view(len(P_features) - N1, -1)
    P_fea_tr = P_features[tr_IND]
    P_fea_te = P_features[te_IND]

    # sample N1 from Q
    np.random.seed(seed=819 * (kk + 9) + N1)
    tr_IND_Q = np.random.choice(len(Q_features), N1, replace=False)
    te_IND_Q = np.delete(np.arange(len(Q_features)), tr_IND_Q)
    Q_tr = Q[tr_IND_Q].view(N1, -1)
    Q_te = Q[te_IND_Q].view(len(Q_features) - N1, -1)
    Q_fea_tr = Q_features[tr_IND_Q]
    Q_fea_te = Q_features[te_IND_Q]

    # Train
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    Dxy_org = Pdist2(P_tr, Q_tr)
    Dxy_fea = Pdist2(P_fea_tr, Q_fea_tr)
    # b_q = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    b_q = Dxy_org.median()
    b_phi = Dxy_fea.median()

    c_epsilon = torch.tensor(1.0).to(device)
    b_q = torch.nn.Parameter(b_q)
    b_phi = torch.nn.Parameter(b_phi)

    c_epsilon.requires_grad = True
    b_q.requires_grad = True
    b_phi.requires_grad = True
    

    optimizer = torch.optim.Adam([c_epsilon]+[b_q]+[b_phi], lr=0.0002)

    Z = torch.cat([P_tr, Q_tr], dim=0)
    Z_fea = torch.cat([P_fea_tr, Q_fea_tr], dim=0)
    # Compute pairwise distances
    pairwise_matrix = torch_distance(Z, Z, norm=2, is_squared=True)
    pairwise_matrix_f = torch_distance(Z_fea, Z_fea, norm=2, is_squared=True)

    for t in range(opt.n_epochs):
        optimizer.zero_grad()
        epsilon = torch.sigmoid(c_epsilon)
        stats, mmd = deep_objective(pairwise_matrix_f, pairwise_matrix, epsilon, b_q, b_phi, N1)
        stats.backward(retain_graph=True)
        optimizer.step()
        if t % 100 == 0:
            print("mmd: ", mmd.item(), "stats: ", -1*stats.item())
    
    H_adaptive = np.zeros(N)
    T_adaptive = np.zeros(N)
    M_adaptive = np.zeros(N)

    np.random.seed(1021*kk + 181 + N1)
    torch.manual_seed(1021*kk + 181 + N1)
    for k in range(N):
        te_IND_P_N1 = np.random.choice(len(P_te), N1, replace=False)
        P_te_N1 = P_te[te_IND_P_N1]
        P_fea_te_N1 = P_fea_te[te_IND_P_N1] 
        
        te_IND_Q_N1 = np.random.choice(len(Q_te), N1, replace=False)
        Q_te_N1 = Q_te[te_IND_Q_N1]
        Q_fea_te_N1 = Q_fea_te[te_IND_Q_N1]

        Z_te = torch.cat([P_te_N1, Q_te_N1], dim=0)
        Z_fea_te = torch.cat([P_fea_te_N1, Q_fea_te_N1], dim=0)
        p_value, th, mmd = mmd_permutation_test(Z_te, Z_fea_te, N1, num_permutations=100, kernel="com2", params=[c_epsilon, b_q, b_phi])
        # print(f"p_value: {p_value}, th: {th}, mmd: {mmd}")
        
        H_adaptive[k] = p_value < alpha
        T_adaptive[k] = th
        M_adaptive[k] = mmd

    print(f"Test power of SAMMD: {H_adaptive.sum() / N}")
    Results[kk] = H_adaptive.sum() / N

    print("Test Power of Baselines (K times): ")
    print(Results)
    print("Average Test Power of SAMMD (K times): ")
    print("SAMMD: ", (Results.sum() / (kk + 1)))
np.save('./Results_CIFAR10_' + str(N1) + 'SAMMD', Results)
    
    