import torch
import numpy as np
import scipy
import sys
import builtins
from torch.autograd import Function

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
        
def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def torch_distance(X, Y, norm=2, max_size=None, matrix=True, is_squared=False):
    if X.dim() == 4:
        # Flatten to (batch_size, channel*height*width)
        X = X.view(X.size(0), -1)
    if Y.dim() == 4:
        # Flatten to (batch_size, channel*height*width)
        Y = Y.view(Y.size(0), -1)

    # Ensure X and Y are at least 2D (handles 1D cases)
    if X.dim() == 1:
        X = X.view(-1, 1)
    if Y.dim() == 1:
        Y = Y.view(-1, 1)

    # Broadcasting the subtraction operation across all pairs of vectors
    diff = X[None, :, :] - Y[:, None, :]
    # print(diff.size())

    if norm == 2:
        # Computing the L2 distance (Euclidean distance)
        if is_squared:
            # If the pairwise matrix is squared
            dist = torch.sum(diff**2, dim=-1)
        else:
            dist = torch.sqrt(torch.sum(diff**2, dim=-1))
    elif norm == 1:
        # Computing the L1 distance (Manhattan distance)
        dist = torch.sum(torch.abs(diff), dim=-1)
    else:
        raise ValueError("Norm must be L1 or L2")

    if max_size:
        dist = dist[:max_size, :max_size]

    if matrix:
        return dist
    else:
        m = dist.shape[0]
        indices = torch.triu_indices(m, m, offset=0)
        return dist[indices[0], indices[1]]

def gaussian_kernel(pairwise_matrix, bandwidth, scale=False, is_squared=True):
    d = pairwise_matrix / bandwidth

    if scale:
        return torch.exp(-(d**2))
    
    if is_squared:
        # If the pairwise matrix is squared
        return torch.exp(-(d/bandwidth) / 2)

    return torch.exp(-(d**2) / 2)

def kernel_matrix(pairwise_matrix, kernel, bandwidth, scale=False):
    if kernel == "gaussian":
        return gaussian_kernel(pairwise_matrix, bandwidth, scale)
    elif kernel == "":
        return None

def mmd_u(K, n, m, is_var=False):
    n_samples = n + m
    # Extract submatrices for XX, YY, and XY
    K_XX = K[:n, :n]
    K_YY = K[n:, n:]
    K_XY = K[:n, n:]

    # Ensure diagonal elements are zero (no self-comparison) for XX and YY
    K_XX.fill_diagonal_(0)
    K_YY.fill_diagonal_(0)

    # Calculate each term of the MMD_u^2
    mmd_u_squared = (K_XX.sum() / (n * (n - 1))) + \
        (K_YY.sum() / (m * (m - 1))) - (2 * K_XY.sum() / (n * m))
    
    if is_var:
        h_matrix = K_XX + K_YY - K_XY - K_XY.t()
        row_means = h_matrix.sum(1) / m
        V1 = torch.dot(row_means, row_means) / m
        V2 = h_matrix.sum() / (n * n)
        variance = 4 * (V1 - V2**2)
        return mmd_u_squared, variance
    return mmd_u_squared

def mmd_u1(K, n, m, is_var=False):
    n_samples = n
    # Extract submatrices for XX, YY, and XY
    K_XX = K[:n, :n]
    K_YY = K[n:, n:]
    K_XY = K[:n, n:]

    # Ensure diagonal elements are zero (no self-comparison) for XX and YY
    K_XX.fill_diagonal_(0)
    K_YY.fill_diagonal_(0)

    # Calculate each term of the MMD_u^2
    # mmd_u_squared = (K_XX.sum() / (n * (n - 1))) + \
    #     (K_YY.sum() / (m * (m - 1))) - (2 * K_XY.sum() / (n * m))
    mmd_u_squared = (K_XX.sum() / (n * (n - 1))) + \
        (K_YY.sum() / (n * (n - 1))) - (2 * K_XY.sum() / (n * n))
    # mmd_u_squared = (K_YY.sum() / (m * (m - 1)))
    
    # mmd_u_squared = (K_XX.sum() / (n * (n - 1))) + \
    #     (K_YY.sum() / (m * (m - 1))) + (2 * K_XY.sum() / (n * m))
    # mmd_u_squared = - (K_XX.sum() / (n * (n - 1))) + \
    #     (K_YY.sum() / (m * (m - 1))) + (2 * K_XY.sum() / (n * m))
    # mmd_u_squared = - (K_XX.sum() / (n * (n - 1))) + (2 * K_XY.sum() / (n * m))
    # mmd_u_squared = (K_YY.sum() / (m * (m - 1))) + (2 * K_XY.sum() / (n * m))
    # mmd_u_squared = -(2 * K_XY.sum() / (n * m))
    # mmd_u_squared = - (K_XX.sum() / (m * (m - 1)))
    
    if is_var:
        h_matrix = K_XX + K_YY - K_XY - K_XY.t()
        row_means = h_matrix.sum(1) / m
        V1 = torch.dot(row_means, row_means) / m
        V2 = h_matrix.sum() / (n * n)
        variance = 4 * (V1 - V2**2)
        return mmd_u_squared, variance
    return mmd_u_squared

def deep_objective(pairwise_matrix_f, pairwise_matrix, epsilon, b_q, b_phi, n_samples, kernel="gaussian"):

    # Compute kernel matrices
    K_q = kernel_matrix(pairwise_matrix, kernel, b_q)
    K_phi = kernel_matrix(pairwise_matrix_f, kernel, b_phi)

    # Compute deep kernel matrix
    K_deep = (1-epsilon) * K_phi * K_q + epsilon * K_q
    # K_deep = 1 - torch.abs(K_q - K_phi)
    # K_deep = torch.max(K_q, K_phi) / (torch.max(K_q, K_phi))
    # K_deep = K_q + K_phi

    tmp = mmd_u(K_deep, n_samples, n_samples, is_var=True)
    mmd_value, mmd_std = tmp[0]+1e-8, torch.sqrt(tmp[1]+1e-8)

    if mmd_std.item() == 0:
        print("Warning: Zero variance in MMD estimate")

    stats = torch.div(-1*mmd_value, mmd_std)

    return stats, mmd_value

def rep(A, model, path):
    if 'cifar' in path:
        n_channel = 3
        img_size = 32
    A_reshaped = A.view(-1, n_channel, img_size, img_size)
    A_rep = model(A_reshaped)
    return A_rep

def batch_cov_einsum(x):
    """
    x: (batch_size, n_samples, n_features)
    cov: (batch_size, n_features, n_features)
    """
    B, N, D = x.shape
    mean = x.mean(dim=1, keepdim=True)  # (B, 1, D)
    x_centered = x - mean
    factor = 1 / (N - 1)

    cov = factor * torch.einsum('bni,bnj->bij', x_centered, x_centered)
    return cov

def log_cov(x, tol=1e-6):
    L, V = torch.linalg.eigh(x)
    L = torch.clamp(L, min=tol)
    log_L = torch.log(L)
    log_Zi = V @ torch.diag(log_L) @ V.t()
    return log_Zi

def inv_sqrt_cov(x, tol=1e-6):
    L, V = torch.linalg.eigh(x)
    L = torch.clamp(L, min=tol)
    inv_sqrt_L = 1 / torch.sqrt(L)
    inv_sqrt_Zi = V @ torch.diag(inv_sqrt_L) @ V.t()
    return inv_sqrt_Zi

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

def mmd_permutation_test4(Z, n_samples, n_per=100, kernel="log_rbf", tol=1e-6):
    # sigma = torch.median(torch_distance(Z, Z, norm=2, is_squared=True)) #XUNYE: use this get identify kernel matrix
    # simga = 1 #XUNYE: 太小的都不行
    sigma = 10
    K = torch.zeros((Z.size(0), Z.size(0)), device=Z.device)
    for i in range(Z.size(0)):

        if kernel == "log_rbf":
            log_Zi = log_cov(Z[i], tol) #XUNYE: 这个绝对没问题
        elif kernel == "airm":
            inv_sqrt_Zi = torch.inverse(sqrtm(Z[i]+1e-6*torch.eye(Z[i].size(0), device=Z.device))) #XUNYE: 这个也绝对没问题

        for j in range(i, Z.size(0)):
            if kernel == "log_rbf":
                log_Zj = log_cov(Z[j], tol)
                # sigma = torch.median(torch_distance(log_Zi, log_Zj, norm=2, is_squared=True)) #XUNYE: this may works, unsure
                kij = torch.exp(-torch.norm(log_Zi - log_Zj)**2 / (2 * sigma **2))
            elif kernel == "airm":
                C121 = inv_sqrt_Zi @ Z[j] @ inv_sqrt_Zi #XUNYE: not identity matrix even if Z[i] == Z[j]
                kij = torch.exp(-torch.norm(log_cov(C121, tol))**2 / (2 * sigma **2))
            elif kernel == "stein":
                logdet_avg = torch.linalg.slogdet((Z[i]+Z[j]) / 2.0)[1]
                logdet_prod = torch.linalg.slogdet(Z[i] @ Z[j])[1] / 2.0
                kij = torch.exp(-sigma * (logdet_avg - logdet_prod)) #XUNYE: logdet_avg - logdet_prod != 0 if Z[i] == Z[j]
            
            K[i, j] = kij
            K[j, i] = kij

    observed_mmd = mmd_u1(K.clone(), n_samples, len(K)-n_samples)
    # print("Observed MMD:", observed_mmd)

    count = 0
    # For each permutation, simply reorder the precomputed kernel matrix
    mmd_ps = []
    for _ in range(n_per):
        perm = torch.randperm(Z.size(0), device=Z.device)
        # Use the permutation to index into K:
        K_perm = K[perm][:, perm]

        # Compute the MMD statistic for the permuted kernel matrix
        perm_mmd = mmd_u1(K_perm, n_samples, len(K)-n_samples)
        # print("Permuted MMD:", perm_mmd)
        mmd_ps.append(perm_mmd)

        if torch.isnan(perm_mmd):
            print('NAN')
            sys.exit(1)
        
        if perm_mmd >= observed_mmd:
            count += 1

    p_value = count / n_per
    
    if torch.isnan(observed_mmd):
        print('NAN')
        sys.exit(1)
    
    return p_value, torch.sort(torch.tensor(mmd_ps))[0][int(n_per * 0.95)], observed_mmd
    