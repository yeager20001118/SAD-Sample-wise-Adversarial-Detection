import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys



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
    mmd_u_squared = (K_XX.sum() / (n * (n - 1))) + \
        (K_YY.sum() / (m * (m - 1))) - (2 * K_XY.sum() / (n * m))
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
    
    L = 100.0
    
    if kernel == "com1":
        b_k = torch.median(pairwise_matrix)
        K_num = gaussian_kernel(pairwise_matrix, b_k)
        K_den = K
        K = K_num / (K_den+L)
    elif kernel == "com2":
        b_k = torch.median(pairwise_matrix)
        K_num = K
        K_den = gaussian_kernel(pairwise_matrix, b_k)
        K = K_num / (K_den+L)
    elif kernel == "com3":
        b_k = torch.median(pairwise_matrix)
        K_org = gaussian_kernel(pairwise_matrix, b_k)
        K_phi = K
        # K = - torch.max(K_org, K_phi) / torch.min(K_org, K_phi)
        # K =  K_org/(K_phi+L) - K_phi/(K_org+L)
        # K =  torch.max(K_org/(K_phi+L), - K_phi/(K_org+L))
        # K1 =  K_org/(K_phi+L)
        # K1 = K1/torch.sum(K1**2)*n_samples**2
        # K2 =  K_phi/(K_org+L)
        # K2 = K2/torch.sum(K2**2)*n_samples**2
        # K = K1 - K2
        
        K =  1 - torch.abs(K_org - K_phi)
        
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

def mmd_permutation_test3(Z, Z_fea, n_samples, num_permutations=100, kernel="deep", params=[1.0]):

    if kernel == "deep":
        c_epsilon, b_q, b_phi = params
    if "com" in kernel:
        c_epsilon, b_q, b_phi = params

    # Compute the pairwise distance matrix for the full data
    pairwise_matrix = torch_distance(Z, Z, norm=2, is_squared=True)
    pairwise_matrix_f = torch_distance(Z_fea, Z_fea, norm=2, is_squared=True)
    L = 0.1

    K_q = gaussian_kernel(pairwise_matrix, b_q).float()
    K_phi = gaussian_kernel(pairwise_matrix_f, b_phi).float()
        
    # # K_q = (K_q - K_q.mean()) / K_q.std()
    # # K_phi = (K_phi - K_phi.mean()) / K_phi.std()
    K_q = (K_q - K_q[:n_samples,:n_samples].mean()) / K_q[:n_samples,:n_samples].std()
    K_phi = (K_phi - K_phi[:n_samples,:n_samples].mean()) / K_phi[:n_samples,:n_samples].std()
    K_q = (K_q - K_q.min()) / (K_q.max() - K_q.min()) + L
    K_phi = (K_phi - K_phi.min()) / (K_phi.max() - K_phi.min()) + L
    # K_q = (K_q - K_q[:n_samples,:n_samples].min()) / (K_q[:n_samples,:n_samples].max() - K_q[:n_samples,:n_samples].min()) + L
    # K_phi = (K_phi - K_phi[:n_samples,:n_samples].min()) / (K_phi[:n_samples,:n_samples].max() - K_phi[:n_samples,:n_samples].min()) + L
    # K_q = K_q - K_q[:n_samples,:n_samples].min() + L
    # K_phi = K_phi - K_phi[:n_samples,:n_samples].min() + L
    # K_q = (K_q - K_q.min()) + L
    # K_phi = (K_phi - K_phi.min()) + L
    mean_q = K_q[:n_samples,:n_samples].mean()
    mean_phi = K_phi[:n_samples,:n_samples].mean()
    K_phi = K_phi * (mean_q / mean_phi)
    
    L = 0
    if kernel == "com1":
        K = K_q / (K_phi+L)
    elif kernel == "com2":
        K = K_phi / (K_q+L)
    elif kernel == "com3":
        K = torch.max(K_q, K_phi)/(torch.min(K_q, K_phi))
        # K = torch.min(K_q, K_phi) / (torch.max(K_q, K_phi)+L)
        # K =  torch.abs(K_q - K_phi)
        # K = 1 - torch.abs(K_q - K_phi)
        # K = torch.max(K_q, K_phi)
        # K = - torch.min(K_q, K_phi)
        # K = K_q
        # K = K_phi
    # K = (1 - c_epsilon) * K_phi * K_q + c_epsilon * K_q

    # Compute the observed MMD
    observed_mmd = mmd_u1(K.clone(), n_samples, n_samples)
    # print("Observed MMD:", observed_mmd)

    count = 0
    # For each permutation, simply reorder the precomputed kernel matrix
    mmd_ps = []
    for _ in range(num_permutations):
        perm = torch.randperm(Z.size(0), device=Z.device)
        # Use the permutation to index into K:
        K_perm = K[perm][:, perm]

        # Compute the MMD statistic for the permuted kernel matrix
        perm_mmd = mmd_u1(K_perm, n_samples, n_samples)
        # print("Permuted MMD:", perm_mmd)
        mmd_ps.append(perm_mmd)

        if torch.isnan(perm_mmd):
            print('NAN')
            sys.exit(1)
        
        if perm_mmd >= observed_mmd:
            count += 1

    p_value = count / num_permutations
    
    if torch.isnan(observed_mmd):
        print('NAN')
        sys.exit(1)
    
    return p_value, torch.sort(torch.tensor(mmd_ps))[0][int(num_permutations * 0.95)], observed_mmd

def mmd_permutation_test2(Z, Z_fea, n_samples, num_permutations=100, kernel="deep", params=[1.0]):

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
    
    b_k = torch.median(pairwise_matrix)
    K_org = K_q
    K_phi = K
    K1 =  -K_org/K_phi
    K1 = K1/torch.sum(K1**2)*n_samples**4
    K2 =  K_phi/K_org
    K2 = K2/torch.sum(K2**2)*n_samples**4
        
    # Compute the observed MMD
    L = 5
    observed_mmd_1 = mmd_u(K1, n_samples, n_samples)
    observed_mmd_2 = mmd_u(K2, n_samples, n_samples)
    observed_mmd = 1/L*torch.log(L*(torch.exp(observed_mmd_1)+torch.exp(observed_mmd_2))/2)
    # print("Observed MMD:", observed_mmd)

    mmd1_ps = []
    mmd2_ps = []
    mmd_ps = []
    for _ in range(num_permutations):
        perm = torch.randperm(Z.size(0), device=Z.device)
        # Use the permutation to index into K:
        K1_perm = K1[perm][:, perm]
        K2_perm = K2[perm][:, perm]

        # Compute the MMD statistic for the permuted kernel matrix
        perm_mmd1 = mmd_u(K1_perm, n_samples, n_samples)
        perm_mmd2 = mmd_u(K2_perm, n_samples, n_samples)
        perm_mmd = 1/L*torch.log(L*(torch.exp(perm_mmd1)+torch.exp(perm_mmd2))/2)
        # print("Permuted MMD:", perm_mmd)
        mmd1_ps.append(perm_mmd1.item())
        mmd2_ps.append(perm_mmd2.item())
        mmd_ps.append(perm_mmd.item())

    p_value1 = (np.sum(np.array(mmd1_ps) >= observed_mmd_1.item()) + 1) / (len(mmd1_ps) + 1)
    p_value2 = (np.sum(np.array(mmd2_ps) >= observed_mmd_2.item()) + 1) / (len(mmd2_ps) + 1)
    p_value = (np.sum(np.array(mmd_ps) >= observed_mmd.item()) + 1) / (len(mmd_ps) + 1)
    
    return p_value, mmd_ps[np.int64(num_permutations*0.95)], observed_mmd