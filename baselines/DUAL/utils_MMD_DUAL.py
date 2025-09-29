import torch
from torch.autograd import Function
import numpy as np
import torch.nn as nn
import scipy
import builtins
import time

# UDF authored by Alex, Jiacheng, Xunye, Yiyi, Zesheng, and Zhijian
builtins.IS_LOG=True
def log(*args, sep=True, **kwargs):
    if builtins.IS_LOG:
        if sep:
            msg = " ".join(map(str, args))
            total_len = 30
            msg_len = len(msg)
            if msg_len >= total_len:
                print(msg)
            else:
                eq_len = total_len - msg_len
                left = eq_len // 2
                right = eq_len - left
                print(f"{'=' * left}{msg}{'=' * right}")
        else:
            print(*args, **kwargs)

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input
sqrtm = MatrixSquareRoot.apply

class DUAL(nn.Module):
    def __init__(self, X, Y, n_bandwidth, k_b_pair, alpha, reg=1e-5, is_cov=True):
        super(DUAL, self).__init__()
        self.X = X
        self.Y = Y
        self.k_b_pair = k_b_pair
        self.kernels = [pair[0] for pair in k_b_pair]
        self.bandwidths = nn.ParameterList()
        self.n_bandwidth = n_bandwidth
        self.is_cov = is_cov
        self.device = X.device
        self.alpha = alpha
        self.reg = reg
        
        for _, bandwidth in k_b_pair:
            self.bandwidths.append(nn.Parameter(bandwidth.clone()))

    def forward():
        pass

    def compute_U_stats(self, X_test=None, Y_test=None, n_per=None):
        device = self.device

        if X_test is not None and Y_test is not None:
            Z = torch.cat([X_test, Y_test], dim=0).to(device)
            n, m = len(X_test), len(Y_test)

            # n = min(len(X_test), len(Y_test))
            # m = len(Y_test)
            # Z = torch.cat([X_test[:n], Y_test[:n]], dim=0)
        else:
            Z = torch.cat([self.X, self.Y], dim=0).to(device)
            n, m = len(self.X), len(self.Y)

            # n = min(len(self.X), len(self.Y))
            # m = len(self.Y)
            # Z = torch.cat([self.X[:n], self.Y[:n]], dim=0)

        dist = torch.cdist(Z, Z, p=2)
        n_b = len(self.bandwidths)
        
        U = []  # U.size() = (c, 1) column vector
        U_b = []  # U_b.size() = (c, n_per)
        for i in range(n_b):
            K = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=dist)
            mmd = mmd_u(K, n, m=m)
            # mmd, h_matrix = mmd_u(K, n, m, need_h=True)
            U.append(mmd)
            if n_per is not None:
                mmd_per = fast_perm(K, n, m, n_per, device)
                U_b.append(mmd_per)

        U = torch.stack(U)
        if n_per is not None:
            U_b = torch.stack(U_b)  # U_b.size() = (c, n_per) matrix
            return U.unsqueeze(1), U_b

        return U.unsqueeze(1)

    def compute_Sigma(self, X_test=None, Y_test=None):
        device = self.device
        n_b = len(self.bandwidths)    
        if not self.is_cov:
            return torch.eye(n_b).to(device, torch.float32)
        
        if X_test is not None and Y_test is not None:
            n = min(len(X_test), len(Y_test))
            m = len(Y_test)
            Z = torch.cat([X_test[:n], Y_test[:n]], dim=0)
        else:
            n = min(len(self.X), len(self.Y))
            m = len(self.Y)
            Z = torch.cat([self.X[:n], self.Y[:n]], dim=0)
        
        indices = torch.randperm(Z.size(0), device=Z.device)
        Z_null = Z[indices]
        
        dist = torch.cdist(Z_null, Z_null, p=2)
        Sigma = torch.zeros(n_b, n_b).to(device, torch.float32)
        C = self.get_C(2, Z_null.size(0))
        for i in range(n_b):
            for j in range(i, n_b):
                K1 = kernel_matrix(self.kernels[i], self.bandwidths[i], dist=dist)
                _, h_matrix1 = mmd_u(K1, n, m, need_h=True)
                K2 = kernel_matrix(self.kernels[j], self.bandwidths[j], dist=dist)
                _, h_matrix2 = mmd_u(K2, n, m, need_h=True)

                mask = torch.triu(torch.ones(n, n), diagonal=1).to(device)

                Sigma[i, j] = C * (h_matrix1 * h_matrix2 * mask).sum()
                Sigma[j, i] = Sigma[i, j]
                
        return Sigma + self.reg * torch.eye(n_b).to(device, torch.float32)
        
    def get_C(self, m, n):
        import math
        return (
            (n**2)
            / (math.comb(n, m))
            * math.comb(m, 2)
            * math.comb(n - m, m - 2)
            / (math.comb(n - 2, m - 2) ** 2)
            / math.comb(n, 2)
        )

    def get_kernel_bandwidth_pairs(self):
        k_b_pair = [(kernel, bandwidth)
                    for kernel, bandwidth in zip(self.kernels, self.bandwidths)]
        print(k_b_pair)
        return k_b_pair

    def check_requires_grad(self):
        # print("Method 1: Check each parameter's requires_grad attribute")
        for name, param in self.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

    # def test_bootstrap(self, n_per):
    #     U_te, U_b = self.compute_U_stats(n_per=n_per)
    #     Sigma = self.compute_Sigma()
    #     n_te = min(len(self.X), len(self.Y)) #TODO
        
    #     T_te = n_te**2 * U_te.T @ torch.inverse(Sigma) @ U_te
    #     M = U_b.T @ torch.inverse(Sigma) @ U_b 
    #     T_b = n_te**2 * torch.diag(M)      

        # p_value = torch.sum(T_b > T_te) / n_per
        # th = torch.sort(T_b)[0][int(n_per * 0.95)]

        # return p_value, th, T_te/(n_te**2*len(self.bandwidths))
    
    def test_permutation(self, n_per):
        time_start = time.time()
        U_te, U_b = self.compute_U_stats(n_per=n_per)
        # time_u = time.time() - time_start
        # print(f"Time for computing U: {time_u}s")
        # print(U_b, U_b.size())
        Sigma = self.compute_Sigma()
        # time_sigma = time.time() - time_start - time_u
        # print(f"Time for computing Sigma: {time_sigma}s")
        n_te = min(len(self.X), len(self.Y))
        
        T_te = n_te**2 * U_te.T @ torch.inverse(Sigma) @ U_te
        M = U_b.T @ torch.inverse(Sigma) @ U_b 
        T_b = n_te**2 * torch.diag(M) 

        p_value = torch.sum(T_b > T_te) / n_per
        # print(f"Time for computing T: {time.time() - time_start - time_u - time_sigma}s")
        th = torch.sort(T_b)[0][int(n_per * 0.95)]

        return p_value, th, T_te/(n_te**2*len(self.bandwidths))

def kernel_matrix(kernel, bandwidth, dist=None):
    # Compute kernel matrix based on the specified kernel
    if "gaussian" in kernel.lower():
        # K_GAUSS(x,y) = exp(-||x-y||²/σ²)
        K = torch.exp(-torch.pow(dist, 2) / torch.pow(bandwidth, 2))
    elif "laplacian" in kernel.lower():
        # K_LAP(x,y) = exp(-||x-y||/σ)
        K = torch.exp(-dist / bandwidth)
    else:
        raise ValueError(
            f"Unknown kernel: {kernel}. Use 'gaussian', 'laplacian' or 'mahalanobis'.")
    return K

def mmd_u(K, n, m, need_h=False):
    """
    Compute the unbiased MMD^2 statistic given the kernel matrix K, if m = n.
    """
    # Extract submatrices for XX, YY, and XY
    K_XX = K[:n, :n]
    K_YY = K[n:, n:]
    K_XY = K[:n, n:]
    # Ensure diagonal elements are zero for XX, YY
    K_XX = K_XX - torch.diag(torch.diag(K_XX))
    K_YY = K_YY - torch.diag(torch.diag(K_YY))
    # Remove diagonal from K_XY (where i=j)
    K_XY[torch.arange(min(n, m)), torch.arange(min(n, m))] = 0
    if need_h:
        h_matrix = K_XX + K_YY - K_XY - K_XY.t()
        # Calculate each term of the MMD_u^2
        mmd_u_squared = h_matrix.sum() / (n * (n - 1))
        return mmd_u_squared, h_matrix
    else:
        mmd_u_squared = (K_XX.sum() / (n * (n - 1))) + \
        (K_YY.sum() / (m * (m - 1))) - (2 * K_XY.sum() / (n * m))
        return mmd_u_squared
        

KERNEL_SET = ['gaussian', 'laplacian']
def generate_kernel_bandwidth(n_bandwidth, X, Y, way):
    assert set(way) <= {'Agg', 'Fuse'}
    k_b_pair = []
    Z = torch.cat([X, Y], dim=0)
    dist = torch.cdist(Z, Z, p=2)
    for i in range(len(n_bandwidth)):
        kernel = KERNEL_SET[i]
        if way[i] == 'Agg':
            bandwidths = get_bandwidth_agg(dist, n_bandwidth[i])
        elif way[i] == 'Fuse':
            bandwidths = get_bandwidth_fuse(dist, n_bandwidth[i])

        for b in bandwidths:
            k_b_pair.append((kernel, b))

    return k_b_pair

def get_bandwidth_agg(dist, n_bandwidth):
    device = dist.device
    median = torch.median(dist).to(device)
    # Compute power sequence
    bandwidths = [2 ** i * median for i in get_power_range(n_bandwidth)]
    return bandwidths

def get_power_range(n):
    if n % 2 == 0:
        # For even n, generate n numbers with 0.5 offset
        start = -(n-1)/2
        return [start + i for i in range(n)]
    else: 
        # For odd n, generate n numbers centered at 0
        start = -(n//2)
        return [start + i for i in range(n)]

def get_bandwidth_fuse(dist, n_bandwidth):
    median = torch.median(dist)
    dist = dist + (dist == 0) * median
    dd = torch.sort(dist)[0].view(-1)
    n = len(dd)
    idx_5 = int(torch.floor(torch.tensor(n * 0.05)).item())
    idx_95 = int(torch.floor(torch.tensor(n * 0.95)).item())
    lambda_min = dd[idx_5] / 2
    lambda_max = dd[idx_95] * 2
    bandwidths = torch.linspace(
        lambda_min, lambda_max, n_bandwidth).to(dist.device)
    return bandwidths

def fast_perm(K, n, m, n_per, device):
    perms = torch.stack([torch.randperm(n+m, device=device) for _ in range(n_per)])
                
    perm_X = perms[:, :n]  # Shape: (n_per, n)
    perm_Y = perms[:, n:]  # Shape: (n_per, m)
    
    # Vectorized submatrix extraction using advanced indexing
    # K_XX_perm: (n_per, n, n)
    K_XX_perm = K[perm_X.unsqueeze(-1), perm_X.unsqueeze(-2)]
    # K_YY_perm: (n_per, m, m) 
    K_YY_perm = K[perm_Y.unsqueeze(-1), perm_Y.unsqueeze(-2)]
    # K_XY_perm: (n_per, n, m)
    K_XY_perm = K[perm_X.unsqueeze(-1), perm_Y.unsqueeze(-2)]
    
    # Remove diagonal elements efficiently
    diag_mask_XX = torch.eye(n, device=device, dtype=torch.bool)
    diag_mask_YY = torch.eye(m, device=device, dtype=torch.bool)
    K_XX_perm[:, diag_mask_XX] = 0
    K_YY_perm[:, diag_mask_YY] = 0
    
    min_nm = min(n, m)
    if min_nm > 0:
        diag_indices = torch.arange(min_nm, device=device)
        K_XY_perm[:, diag_indices, diag_indices] = 0
    
    mmd_per = (K_XX_perm.sum(dim=(1,2)) / (n * (n - 1))) + \
                (K_YY_perm.sum(dim=(1,2)) / (m * (m - 1))) - \
                (2 * K_XY_perm.sum(dim=(1,2)) / (n * m))

    return mmd_per