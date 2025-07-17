import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

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

def deep_objective(pairwise_matrix_f, pairwise_matrix, epsilon, b_q, b_phi, n_samples, kernel="gaussian"):

    # Compute kernel matrices
    K_q = kernel_matrix(pairwise_matrix, kernel, b_q)
    K_phi = kernel_matrix(pairwise_matrix_f, kernel, b_phi)

    # Compute deep kernel matrix
    K_deep = (1-epsilon) * K_phi * K_q + epsilon * K_q

    tmp = mmd_u(K_deep, n_samples, n_samples, is_var=True)
    mmd_value, mmd_std = tmp[0]+1e-8, torch.sqrt(tmp[1]+1e-8)

    if mmd_std.item() == 0:
        print("Warning: Zero variance in MMD estimate")

    stats = torch.div(-1*mmd_value, mmd_std)

    return stats, mmd_value
    
def mmd_permutation_test(Z, Z_fea, n_samples, num_permutations=100, kernel="deep", params=[1.0]):

    if kernel == "deep":
        c_epsilon, b_q, b_phi = params
    if "com" in kernel:
        c_epsilon, b_q, b_phi = params

    # Compute the pairwise distance matrix for the full data
    pairwise_matrix = torch_distance(Z, Z, norm=2, is_squared=True)
    pairwise_matrix_f = torch_distance(Z_fea, Z_fea, norm=2, is_squared=True)

    epsilon = torch.sigmoid(c_epsilon)
    K_q = gaussian_kernel(pairwise_matrix, b_q)
    K_phi = gaussian_kernel(pairwise_matrix_f, b_phi)
    K = (1 - epsilon) * K_phi * K_q + epsilon * K_q
    
    if kernel == "com1":
        b_k = torch.median(pairwise_matrix)
        K_num = gaussian_kernel(pairwise_matrix, b_k)
        K_den = K
        K = K_num / K_den
    elif kernel == "com2":
        b_k = torch.median(pairwise_matrix)
        K_num = K
        K_den = gaussian_kernel(pairwise_matrix, b_k)
        K = K_num / K_den
    elif kernel == "com3":
        b_k = torch.median(pairwise_matrix)
        K_org = gaussian_kernel(pairwise_matrix, b_k)
        K_phi = K
        K = torch.max(K_org, K_phi) / torch.min(K_org, K_phi)
        
    # Compute the observed MMD
    observed_mmd = mmd_u(K, n_samples, n_samples)
    # print("Observed MMD:", observed_mmd)

    count = 0
    # For each permutation, simply reorder the precomputed kernel matrix
    mmd_ps = []
    for _ in range(num_permutations):
        perm = torch.randperm(Z.size(0), device=Z.device)
        # Use the permutation to index into K:
        K_perm = K[perm][:, perm]

        # Compute the MMD statistic for the permuted kernel matrix
        perm_mmd = mmd_u(K_perm, n_samples, n_samples)
        # print("Permuted MMD:", perm_mmd)
        mmd_ps.append(perm_mmd)

        if kernel == "com1" or kernel == "com3":
            if perm_mmd <= observed_mmd:
                count += 1
        else:
            if perm_mmd >= observed_mmd:
                count += 1

    p_value = count / num_permutations
    
    if kernel == "com1" or kernel == "com3":
        return p_value, mmd_ps[np.int64(num_permutations*0.05)], observed_mmd
    else:
        return p_value, mmd_ps[np.int64(num_permutations*0.95)], observed_mmd