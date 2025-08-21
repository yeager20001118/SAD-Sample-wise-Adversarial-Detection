import numpy as np
import torch
import time
import sys

# UDF authored by Alex, Jiacheng, Xunye, Yiyi, Zesheng, and Zhijian
sys.path.append('/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection')
from models import *
from exp.dataloader import load_data
from baselines.SAMMD.utils_SAMMD import *


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def train_SAMMD(path, N1, rs, check, model, N_epoch, lr, ref):
    device = check_device()
    model = model.to(device)
    np.random.seed(rs)
    torch.manual_seed(rs)

    log("path: ", path)
    (P, Q), (P_rep, Q_rep) = load_data(path, N1, rs, check, model, ref = ref, is_test=False)
    Dxy_org = Pdist2(P, Q)
    Dxy_rep = Pdist2(P_rep, Q_rep)
    b_q = Dxy_org.mean()
    b_phi = Dxy_rep.median()

    c_epsilon = torch.tensor(1.0).to(device)
    b_q = torch.nn.Parameter(b_q)
    b_phi = torch.nn.Parameter(b_phi)

    c_epsilon.requires_grad = True
    b_q.requires_grad = True
    b_phi.requires_grad = True

    optimizer = torch.optim.Adam([c_epsilon]+[b_q]+[b_phi], lr=lr)

    Z = torch.cat([P, Q], dim=0)
    Z_fea = torch.cat([P_rep, Q_rep], dim=0)
    pairwise_matrix = torch_distance(Z, Z, norm=2, is_squared=True)
    pairwise_matrix_f = torch_distance(Z_fea, Z_fea, norm=2, is_squared=True)
    
    epoch = 0
    for epoch in range(N_epoch):
        optimizer.zero_grad()
        epsilon = torch.sigmoid(c_epsilon)
        stats, mmd = deep_objective(pairwise_matrix_f, pairwise_matrix, epsilon, b_q, b_phi, N1)
        stats.backward(retain_graph=True)
        optimizer.step()
        if epoch % 100 == 0:
            log("mmd: ", mmd.item(), "stats: ", -1*stats.item())
    log("end training epoch {} with mmd: {} and stats: {}".format(epoch+1, mmd.item(), -1*stats.item()), sep=True) if epoch > 0 else None
    
    return [c_epsilon, b_q, b_phi]

def SAMMD(path, N1, rs, check, model_params, n_test, n_per, alpha, ref):
    device = check_device()
    model, c_epsilon, b_q, b_phi = model_params
    model, c_epsilon, b_q, b_phi = model.to(device), c_epsilon.to(device), b_q.to(device), b_phi.to(device)
    np.random.seed(rs)
    torch.manual_seed(rs)

    (P, Q), (P_rep, Q_rep) = load_data(path, N1, rs, check, model, ref = ref)

    H_SAMMD = np.zeros(n_test)
    T_SAMMD = np.zeros(n_test)
    M_SAMMD = np.zeros(n_test)

    np.random.seed(rs*1021 + N1)
    test_time = 0
    for k in range(n_test):
        P_idx = np.random.choice(len(P), N1, replace=False)
        P_te = P[P_idx]
        P_rep_te = P_rep[P_idx]

        Q_idx = np.random.choice(len(Q), N1, replace=False)
        Q_te = Q[Q_idx]
        Q_rep_te = Q_rep[Q_idx]

        # log("P_te.shape: {}, P_rep_te.shape: {}, Q_te.shape: {}, Q_rep_te.shape: {}".format(P_te.shape, P_rep_te.shape, Q_te.shape, Q_rep_te.shape))

        Z_te = torch.cat([P_te, Q_te], dim=0)
        Z_rep_te = torch.cat([P_rep_te, Q_rep_te], dim=0)
        start_time = time.time()
        p_value, th, mmd = mmd_permutation_test(Z_te, Z_rep_te, N1, n_per=n_per, params=[c_epsilon,b_q, b_phi])
        test_time += time.time() - start_time
        # print(f"p_value: {p_value}, th: {th}, mmd: {mmd}")
        H_SAMMD[k] = not (p_value < alpha)
        T_SAMMD[k] = th
        M_SAMMD[k] = mmd

    log("SAMMD avg test time: ", test_time/n_test, "s")

    return H_SAMMD, T_SAMMD, M_SAMMD, test_time/n_test

