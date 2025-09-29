import numpy as np
import torch
import time
import sys

# UDF authored by Alex, Jiacheng, Xunye, Yiyi, Zesheng, and Zhijian
sys.path.append('/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection')
from models import *
from exp.dataloader import load_data, check_device
from baselines.SAD.utils_SAD import *

def SAD(path, N1, N_ip, rs, check, model, kernel, n_test, n_per, alpha, ref):
    device = check_device()
    # model, c_epsilon, b_q, b_phi = model_params
    # model, c_epsilon, b_q, b_phi = model.to(device), c_epsilon.to(device), b_q.to(device), b_phi.to(device)
    model = model.to(device)
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    (P, Q), (_, _) = load_data(path, N1, rs, check, model, ref = ref, class_idx = 0) #XUNYE: class_idx 从0-9，只load cifar10的10个类，输入None就是随机类

    H_SAD = np.zeros(n_test)
    T_SAD = np.zeros(n_test)
    M_SAD = np.zeros(n_test)

    b_x = 1 # std of generating the noise for x
    b_y = 1
    n_perturb = 10

    np.random.seed(rs*1021 + N1)
    test_time = 0
    for k in range(n_test):

        ##################### 0.06s in 4090 ###########################
        #############XUNYE: Sigma_x and Sigma_y 都检查过了，没问题######
        # Q_idx = np.random.choice(len(Q), N1, replace=False) #TODO
        Q_idx = np.random.choice(len(Q), N_ip, replace=False)
        Q_te = Q[Q_idx]

        P_idx = np.random.choice(len(P), N1, replace=False)
        P_te = P[P_idx]

        gaussian_noise = torch.normal(mean=0.0, std=b_x, size=(P_te.size(0), n_perturb, *P_te[0].shape)).to(device)
        P_te_expanded = P_te.unsqueeze(1).expand(-1, n_perturb, -1)
        all_noisy = P_te_expanded + gaussian_noise # (N, n_perturb, 3072)
        phi_x_all = rep(all_noisy, model, path).view(P_te.size(0), n_perturb, -1) # (N, n_perturb, 64)
        Sigma_x = batch_cov_einsum(phi_x_all)
        
        gaussian_noise = torch.normal(mean=0.0, std=b_y, size=(Q_te.size(0), n_perturb, *Q_te[0].shape)).to(device)
        Q_te_expanded = Q_te.unsqueeze(1).expand(-1, n_perturb, -1)
        all_noisy = Q_te_expanded + gaussian_noise
        phi_y_all = rep(all_noisy, model, path).view(P_te.size(0), n_perturb, -1)
        Sigma_y = batch_cov_einsum(phi_y_all)

        # print(Sigma_x[0].size(), torch.allclose(Sigma_x[0], Sigma_x[0].T, atol=1e-6))
        # print(torch.linalg.eigh(Sigma_x[0]))
        # sys.exit()
        #################################################################

        Z_te = torch.cat([Sigma_x, Sigma_y], dim=0)
        start_time = time.time()
        p_value, th, mmd = mmd_permutation_test4(Z_te, N1, n_per=n_per, kernel=kernel)
        test_time += time.time() - start_time

        H_SAD[k] = not (p_value < alpha) if ref == "ref" else (p_value < alpha)
        T_SAD[k] = th
        M_SAD[k] = mmd
        print("p_value: ", p_value)

    log("SAD avg test time: ", test_time/n_test, "s") # average 1s per test

    return H_SAD, T_SAD, M_SAD, test_time/n_test