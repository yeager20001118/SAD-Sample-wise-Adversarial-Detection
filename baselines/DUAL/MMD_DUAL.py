import numpy as np
import torch
import time
import sys

# UDF authored by Alex, Jiacheng, Xunye, Yiyi, Zesheng, and Zhijian
sys.path.append('/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection')
from models import *
from exp.dataloader import load_data, check_device
from baselines.DUAL.utils_MMD_DUAL import *

def DUAL_no_train(path, N1, N_ip, rs, check, model, n_test, n_per, alpha, n_bandwidth, reg, way, is_cov, ref):
    device = check_device()
    model = model.to(device)
    np.random.seed(rs)
    torch.manual_seed(rs)
    
    # log("path: ", path)
    (P_tr, Q_tr), (P_rep_tr, Q_rep_tr) = load_data(path, N1, rs, check, model, ref = ref, is_test=False)

    H_DUAL = np.zeros(n_test)
    T_DUAL = np.zeros(n_test)
    P_DUAL = np.zeros(n_test)
    
    (P, Q), (P_rep, Q_rep) = load_data(path, N1, rs, check, model=model, ref=ref)
    test_time = 0
    # test by DUAL
    for k in range(n_test):
        # Q_idx = np.random.choice(len(Q), N1, replace=False) #TODO
        Q_idx = np.random.choice(len(Q), N_ip, replace=False)
        Q_te = Q_rep[Q_idx]

        P_idx = np.random.choice(len(P), N1, replace=False)
        P_te = P_rep[P_idx]
        
        # Q_te = torch.cat([Q_tr, Q_te], dim=0)
        # P_te = torch.cat([P_tr, P_te], dim=0)
        # Q_te = torch.cat([Q_rep_tr, Q_te], dim=0)
        P_te = torch.cat([P_rep_tr, P_te], dim=0)

        k_b_pair = generate_kernel_bandwidth(
            n_bandwidth,
            P_te, Q_te, way
        )

        model_DUAL = DUAL(P_te, Q_te, n_bandwidth, k_b_pair, alpha, reg, is_cov).to(device, torch.float32)
        state_time = time.time()
        p_value, th, _ = model_DUAL.test_permutation(n_per)
        test_time += time.time() - state_time

        H_DUAL[k] = not (p_value < alpha)
        T_DUAL[k] = th
        P_DUAL[k] = p_value
        # break

    # uncomment this if you no time logging needed
    if builtins.DUAL_TIME_LOG == 0:
        log("DUAL avg test time: ", test_time/n_test, "s")
        builtins.DUAL_TIME_LOG += 1

    return H_DUAL, T_DUAL, P_DUAL, test_time/n_test