from typing import Any
import torch
import numpy as np
import scipy
import sys
import os
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

def compute_projection_matrix(model, path, device, reduction_method, reduction_dim, n_perturb, std, calibration_data=None):
    """
    Compute projection matrix from calibration data (independent of test data).
    This prevents data leakage by learning the projection from separate data.

    The projection matrix is cached and reused across runs with the same configuration.
    If cached projection exists, calibration_data is not needed.

    Args:
        model: neural network model
        path: dataset path
        device: torch device
        reduction_method: 'pca', 'random', 'svd', or None
        reduction_dim: target dimension (int, float, or None for auto)
        n_perturb: number of perturbations to estimate feature space
        std: standard deviation for perturbations
        calibration_data: calibration samples (only needed if cache doesn't exist)

    Returns:
        projection_matrix: (feature_dim, target_dim) projection matrix, or None
        target_dim: the actual target dimension, or None
    """
    if reduction_method is None:
        return None, None

    # Create cache directory and filename
    cache_dir = os.path.join(builtins.CLEAN_PATH, "projection_matrix")
    os.makedirs(cache_dir, exist_ok=True)

    # Extract dataset identifier from path
    dataset_name = builtins.CLEAN_PATH.split('/')[-1]
    if not dataset_name:
        dataset_name = "unknown"

    # Cache filename includes all parameters that affect the projection
    reduction_dim_str = f"{reduction_dim}" if reduction_dim is not None else "auto"
    cache_file = os.path.join(cache_dir, f"{dataset_name}_{reduction_method}_{reduction_dim_str}_proj_{n_perturb}_{np.round(std, 4)}.pt")

    # Check if cached projection matrix exists
    if os.path.exists(cache_file):
        log(f"Loading cached projection matrix from {cache_file}")
        cached_data = torch.load(cache_file, map_location=device, weights_only=False)
        projection_matrix = cached_data['projection_matrix'].to(device)
        target_dim = cached_data['target_dim']
        feature_dim = cached_data['feature_dim']
        log(f"Loaded projection: {feature_dim} -> {target_dim}")
        return projection_matrix, target_dim

    # Cache doesn't exist, need calibration data to compute projection
    if calibration_data is None:
        raise ValueError(
            f"Projection matrix cache not found at {cache_file}. "
            "Please provide calibration_data to compute the projection matrix."
        )

    log("Computing projection matrix from calibration data...")

    n_samples = calibration_data.size(0)
    chunk_size = 10
    phi_calib_list = []
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        calib_chunk = calibration_data[start_idx:end_idx].to(device)

        log(f"Processing chunk {start_idx//chunk_size + 1}/{(n_samples + chunk_size - 1)//chunk_size}")

        gaussian_noise = torch.normal(mean=0.0, std=std, size=(calib_chunk.size(0), n_perturb, *calib_chunk[0].shape)).to(device)
        calib_expanded = calib_chunk.unsqueeze(1).expand(-1, n_perturb, -1)
        all_noisy = calib_expanded + gaussian_noise # (chunk_size, n_perturb, 3072)

        phi_chunk = rep(all_noisy, model, path).view(calib_chunk.size(0), n_perturb, -1)
        phi_calib_list.append(phi_chunk.cpu())

        del calib_chunk, gaussian_noise, all_noisy, phi_chunk
        torch.cuda.empty_cache()

    phi_calib = torch.cat(phi_calib_list, dim=0).to(device)
    del phi_calib_list

    feature_dim = phi_calib.shape[2]

    # Determine target dimension using heuristics if not provided
    if reduction_dim is None:
        # Heuristic-based dimension selection
        if feature_dim <= 50:
            log(f"Feature dim {feature_dim} <= 50, skipping dimension reduction")
            return None, None
        elif feature_dim <= 128:
            reduction_dim = int(feature_dim * 0.75)
        elif feature_dim <= 512:
            reduction_dim = max(100, int(feature_dim ** 0.5 * 10))
        elif feature_dim <= 2048:
            reduction_dim = max(128, int(feature_dim ** 0.5 * 8))
        else:
            reduction_dim = min(256, max(200, int(feature_dim ** 0.5 * 6)))
        log(f"Auto-selected target_dim={reduction_dim} for feature_dim={feature_dim}")
    elif isinstance(reduction_dim, float) and 0 < reduction_dim < 1:
        reduction_dim = max(1, int(feature_dim * reduction_dim))
        log(f"Fractional reduction: target_dim={reduction_dim} ({reduction_dim/feature_dim*100:.1f}% of {feature_dim})")
    else:
        reduction_dim = int(reduction_dim)

    target_dim = min(reduction_dim, feature_dim)

    if target_dim >= feature_dim:
        log(f"Target dim {target_dim} >= feature dim {feature_dim}, no reduction needed")
        return None, None

    log(f"Computing {reduction_method} projection: {feature_dim} -> {target_dim}")

    # Reshape for projection computation
    features_flat = phi_calib.reshape(-1, feature_dim)

    if reduction_method == 'pca':
        mean = features_flat.mean(dim=0, keepdim=True)
        centered = features_flat - mean
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        variance_explained = (S ** 2) / (S ** 2).sum()
        cumsum_variance = torch.cumsum(variance_explained, dim=0)
        log(f"PCA: {target_dim} components explain {cumsum_variance[target_dim-1].item()*100:.1f}% variance")

        projection_matrix = Vt[:target_dim, :].t()  # (feature_dim, target_dim)

    elif reduction_method == 'svd':
        _, S, Vt = torch.linalg.svd(features_flat, full_matrices=False)
        log(f"SVD: singular value ratio {S[0].item()/S[target_dim-1].item():.2f}")
        projection_matrix = Vt[:target_dim, :].t()  # (feature_dim, target_dim)

    elif reduction_method == 'random':
        projection_matrix = torch.randn(feature_dim, target_dim, device=device, dtype=features_flat.dtype)
        projection_matrix = projection_matrix / torch.sqrt(torch.tensor(target_dim, dtype=features_flat.dtype))
        log(f"Random projection: {feature_dim} -> {target_dim}")

    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}")

    # Save projection matrix to cache
    log(f"Saving projection matrix to {cache_file}")
    torch.save({
        'projection_matrix': projection_matrix.cpu(),
        'target_dim': target_dim,
        'feature_dim': feature_dim,
        'reduction_method': reduction_method,
        'reduction_dim': reduction_dim
    }, cache_file)

    # Clean up
    del phi_calib, features_flat
    torch.cuda.empty_cache()

    return projection_matrix, target_dim

def Pdist(x, y, p, chunk_size=50):
    """compute the paired distance between x and y."""
    # print(x.shape, y.shape, p)
    # sys.exit()
    if y is None:
        y = x
    x = x.reshape(x.size(0), -1)
    y = y.reshape(y.size(0), -1)

    n, d = x.shape
    m = y.shape[0]

    # If data is small enough, use original implementation
    # Threshold based on memory: if n*m*d would create tensor > 1GB
    if n * m * d < 250_000_000:  # ~1GB for float32
        if p == float('inf'):
            diff = x.unsqueeze(1) - y.unsqueeze(0)     # (n,m,d)
            D = diff.abs().amax(dim=-1)                # (n,m)
        elif p >= 1:
            diff = x.unsqueeze(1) - y.unsqueeze(0)      # (n, m, d)
            D = (diff.abs() ** p).sum(dim=-1) ** (1.0/p)
        else:
            raise ValueError("p must be >= 1 (or float('inf'))")
        return D

    # For large tensors, compute in chunks to save memory
    D = torch.zeros((n, m), device=x.device, dtype=x.dtype)

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        x_chunk = x[i:end_i]  # (chunk_size, d)

        for j in range(0, m, chunk_size):
            end_j = min(j + chunk_size, m)
            y_chunk = y[j:end_j]  # (chunk_size, d)

            if p == float('inf'):
                diff = x_chunk.unsqueeze(1) - y_chunk.unsqueeze(0)  # (chunk_i, chunk_j, d)
                D[i:end_i, j:end_j] = diff.abs().amax(dim=-1)
            elif p >= 1:
                diff = x_chunk.unsqueeze(1) - y_chunk.unsqueeze(0)  # (chunk_i, chunk_j, d)
                D[i:end_i, j:end_j] = (diff.abs() ** p).sum(dim=-1) ** (1.0/p)
            else:
                raise ValueError("p must be >= 1 (or float('inf'))")

            del diff
            if x.is_cuda:
                torch.cuda.empty_cache()

    return D

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
        (K_YY.sum() / (n * (n - 1))) - (2 * K_XY.sum() / (m * n))


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
    if 'imagenet' in path:
        n_channel = 3
        img_size = 224
    
    with torch.no_grad():  # Prevent gradient accumulation
        A_reshaped = A.view(-1, n_channel, img_size, img_size)
        A_rep = model(A_reshaped)
        
        # Clear intermediate variables
        del A_reshaped
        torch.cuda.empty_cache()
        
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
    if x.dim() == 2:
        L, V = torch.linalg.eigh(x)
        L = torch.clamp(L, min=tol)
        log_L = torch.log(L)
        log_Zi = V @ torch.diag(log_L) @ V.t()
        return log_Zi
    elif x.dim() == 3:
        L, V = torch.linalg.eigh(x)  # L: (batch, d), V: (batch, d, d)
        L = torch.clamp(L, min=tol)
        log_L = torch.log(L)
        log_diag = torch.diag_embed(log_L)  # (batch, d, d)
        log_Zi = V @ log_diag @ V.transpose(-2, -1)
        return log_Zi
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}. Expected 2 or 3.")

def inv_sqrt_cov(x, tol=1e-6):
    L, V = torch.linalg.eigh(x)
    L = torch.clamp(L, min=tol)
    inv_sqrt_L = 1 / torch.sqrt(L)
    inv_sqrt_Zi = V @ torch.diag(inv_sqrt_L) @ V.t()
    return inv_sqrt_Zi

def sigma_from_perturb(X_te, std, n_perturb, model, path, device, where=None, chunk_size=10,
                       projection_matrix=None, reduction_dim=None, reduction_method=None, use_logits=False):
    """
    Compute covariance matrices from perturbed samples with optional dimension reduction.

    IMPORTANT: To prevent data leakage, projection_matrix should be pre-computed from
    independent calibration data using compute_projection_matrix(), NOT from test data.

    Args:
        X_te: test samples
        std: standard deviation for Gaussian noise
        n_perturb: number of perturbations per sample
        model: neural network model
        path: dataset path (for determining image size)
        device: torch device
        where: cache identifier
        chunk_size: process samples in chunks to save memory
        projection_matrix: PRE-COMPUTED projection matrix from calibration data (REQUIRED for fair testing)
                          Use compute_projection_matrix() to derive this from independent data
        reduction_dim: actual target dimension (should match projection_matrix.shape[1] if provided)
        reduction_method: 'pca', 'svd', 'random', or None (used for cache filename)
        use_logits: if True, use logit layer; otherwise use last representation layer

    Returns:
        Sigma_x: covariance matrices (n_samples, reduced_dim, reduced_dim)

    Examples:
        # CORRECT: Pre-compute projection from calibration data (no leakage)
        proj_matrix, target_dim = compute_projection_matrix(
            model, path, device, 'pca', None, n_perturb, sigma_un, calibration_data=data_sample
        )
        P_sigma = sigma_from_perturb(P, sigma_un, n_perturb, model, path, device, "P",
                                     projection_matrix=proj_matrix, reduction_dim=target_dim,
                                     reduction_method='pca')
        Q_sigma = sigma_from_perturb(Q, sigma_un, n_perturb, model, path, device, "Q",
                                     projection_matrix=proj_matrix, reduction_dim=target_dim,
                                     reduction_method='pca')

        # INCORRECT: No projection matrix (backward compatible but no reduction)
        P_sigma = sigma_from_perturb(P, sigma_un, n_perturb, model, path, device, "P")
    """
    # Create filename for caching
    cache_dir = os.path.join(builtins.CLEAN_PATH, "Sigma_matrix")
    os.makedirs(cache_dir, exist_ok=True)

    # Include reduction parameters in cache filename
    # Format: "pca_50" if reduction, "full" otherwise
    has_reduction = projection_matrix is not None
    if has_reduction:
        reduction_str = f"{reduction_method}_{reduction_dim}"
    else:
        reduction_str = "full"
    logit_str = "logits" if use_logits else "rep"
    cache_file = os.path.join(cache_dir, f"{where}_{len(X_te)}_{std}_{n_perturb}_{reduction_str}_{logit_str}.pt")

    # Check if cached file exists
    if os.path.exists(cache_file):
        log(f"Loading Sigma matrix from {cache_file}")
        return torch.load(cache_file, map_location=device, weights_only=True)

    log(f"Computing Sigma matrix for {len(X_te)} samples with chunk_size={chunk_size}")
    if has_reduction:
        log(f"Using pre-computed projection matrix: {projection_matrix.shape[0]} -> {reduction_dim}")
    else:
        log("No dimension reduction (projection_matrix=None)")

    # Process in chunks to avoid CUDA memory issues
    n_samples = X_te.size(0)
    sigma_list = []

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        X_chunk = X_te[start_idx:end_idx]

        log(f"Processing chunk {start_idx//chunk_size + 1}/{(n_samples + chunk_size - 1)//chunk_size}")

        # Generate noise for current chunk
        gaussian_noise = torch.normal(mean=0.0, std=std, size=(X_chunk.size(0), n_perturb, *X_chunk[0].shape)).to(device)
        P_te_expanded = X_chunk.unsqueeze(1).expand(-1, n_perturb, -1)
        all_noisy = P_te_expanded + gaussian_noise # (chunk_size, n_perturb, 3072)

        phi_x_all = rep(all_noisy, model, path).view(X_chunk.size(0), n_perturb, -1) # (chunk_size, n_perturb, feature_dim)

        # Apply pre-computed projection if provided
        if projection_matrix is not None:
            batch_size, n_samples_chunk, feature_dim = phi_x_all.shape
            phi_x_flat = phi_x_all.reshape(-1, feature_dim)
            phi_x_reduced = phi_x_flat @ projection_matrix  # (batch * n_samples, target_dim)
            phi_x_all = phi_x_reduced.reshape(batch_size, n_samples_chunk, reduction_dim)

        # Compute covariance
        Sigma_chunk = batch_cov_einsum(phi_x_all) # (chunk_size, reduced_dim, reduced_dim)
        sigma_list.append(Sigma_chunk.cpu())

        del gaussian_noise, P_te_expanded, all_noisy, phi_x_all, Sigma_chunk
        torch.cuda.empty_cache()

    Sigma_x = torch.cat(sigma_list, dim=0).to(device)

    # Save to cache file
    log(f"Saving Sigma matrix to {cache_file}")
    torch.save(Sigma_x.cpu(), cache_file)

    return Sigma_x

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

def MD_kernel_matrix(Z, b, kernel, chunk_size=1000):
    n = Z.size(0)
    if n <= chunk_size:
        if kernel == "Gaussian":
            pw_matrix = Pdist(Z, Z, 2)
        elif kernel == "Laplacian":
            pw_matrix = Pdist(Z, Z, 1)
        K = torch.exp(-pw_matrix / b)
        return K
    else:
        K = torch.zeros((n, n), device=Z.device)
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            Z_chunk_i = Z[i:end_i]
            for j in range(0, n, chunk_size):
                end_j = min(j + chunk_size, n)
                Z_chunk_j = Z[j:end_j]
                if kernel == "Gaussian":
                    pw_matrix = Pdist(Z_chunk_i, Z_chunk_j, 2)
                elif kernel == "Laplacian":
                    pw_matrix = Pdist(Z_chunk_i, Z_chunk_j, 1)
                K[i:end_i, j:end_j] = torch.exp(-pw_matrix / b)
                del pw_matrix
                torch.cuda.empty_cache()
        return K

def VD_kernel_matrix(Z, b, kernel, chunk_size=1000):
    n = Z.size(0)
    if n <= chunk_size:
        if kernel == "Gaussian":
            pw_matrix = Pdist(Z, Z, 2)
        elif kernel == "Laplacian":
            pw_matrix = Pdist(Z, Z, 1)
        K = torch.exp(-pw_matrix / b)
        return K
    else:
        K = torch.zeros((n, n), device=Z.device)
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            Z_chunk_i = Z[i:end_i]
            for j in range(0, n, chunk_size):
                end_j = min(j + chunk_size, n)
                Z_chunk_j = Z[j:end_j]
                if kernel == "Gaussian":
                    pw_matrix = Pdist(Z_chunk_i, Z_chunk_j, 2)
                elif kernel == "Laplacian":
                    pw_matrix = Pdist(Z_chunk_i, Z_chunk_j, 1)
                K[i:end_i, j:end_j] = torch.exp(-pw_matrix / b)
                del pw_matrix
                torch.cuda.empty_cache()
        return K

def CD_kernel_matrix(Z, b, kernel, tol=1e-6, chunk_size=500, Z_dist_matrix=None, log_Z_dist_matrix=None):
    """
    Compute kernel matrix for covariance distributions.

    Args:
        Z: covariance matrices (n, d, d)
        b: kernel bandwidth
        kernel: kernel type ('Gaussian', 'Laplacian', 'log_rbf')
        tol: tolerance for eigenvalue clamping in log_cov
        chunk_size: chunk size for large matrices
        Z_dist_matrix: pre-computed distance matrix for Z (optional, for 'Gaussian'/'Laplacian')
        log_Z_dist_matrix: pre-computed distance matrix for log(Z) (optional, for 'log_rbf')

    Returns:
        K: kernel matrix (n, n)
    """
    n = Z.size(0)

    # Use pre-computed distance matrix if available
    if kernel in ["Gaussian", "Laplacian"] and Z_dist_matrix is not None:
        pw_matrix = Z_dist_matrix
        K = torch.exp(-pw_matrix / b)
        return K
    elif kernel == "log_rbf" and log_Z_dist_matrix is not None:
        pw_matrix = log_Z_dist_matrix
        K = torch.exp(-pw_matrix / b)
        return K

    # Original implementation when no pre-computed distance matrix is provided
    if n <= chunk_size:
        if kernel == "Gaussian":
            pw_matrix = Pdist(Z, Z, 2)
        elif kernel == "Laplacian":
            pw_matrix = Pdist(Z, Z, 1)
        elif kernel == "log_rbf":
            log_covs = torch.stack([log_cov(Z[i], tol) for i in range(n)], dim=0)  # (n, d, d)
            pw_matrix = Pdist(log_covs, log_covs, 2)
        K = torch.exp(-pw_matrix / b)
        return K
    else:
        K = torch.zeros((n, n), device=Z.device)
        if kernel in ["Gaussian","Laplacian"]:
            log_covs = Z
        elif kernel in ["log_rbf"]:
            log_covs = torch.stack([log_cov(Z[i], tol) for i in range(n)], dim=0)  # (n, d, d)
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            log_chunk_i = log_covs[i:end_i]  # (chunk_size, d, d)
            for j in range(0, n, chunk_size):
                end_j = min(j + chunk_size, n)
                log_chunk_j = log_covs[j:end_j]  # (chunk_size, d, d)
                if kernel in ["Gaussian","log_rbf"]:
                    pw_matrix = Pdist(log_chunk_i, log_chunk_j, 2)
                elif kernel in ["Laplacian"]:
                    pw_matrix = Pdist(log_chunk_i, log_chunk_j, 1)
                K[i:end_i, j:end_j] = torch.exp(-pw_matrix / b)
                del pw_matrix
                torch.cuda.empty_cache()
        return K

def get_fuse_km(Z_rep, Z_sigma, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD, stat_name):
    if stat_name == "mean":
        return MD_kernel_matrix(Z_rep, b_MMD, kernel_MMD)
    elif stat_name == "variance":
        return VD_kernel_matrix(Z_rep, b_VD, kernel_VD)
    elif stat_name == "uncertainty":
        return CD_kernel_matrix(Z_sigma, b_CD, kernel_CD)

def vd_u1(K, n, m):
    # mean(diag(K_XX)) = mean(diag(K_YY)) = 1 
    
    K.fill_diagonal_(0)
    K_XX = K[:n, :n]
    K_YY = K[n:, n:]
    
    return (K_YY.sum() / (m * (m - 1)) - K_XX.sum() / (n * (n - 1)) )**2

def fuse_by_name(K, N1, N2, stat_name):
    if stat_name in ["mean", "uncertainty"]:
        return mmd_u1(K.clone(), N1, N2)
    elif stat_name in ["variance"]:
        return vd_u1(K.clone(), N1, N2)

def get_fuse_stats(Z_rep, Z_sigma, N1, stat_names, n_per, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD):
    fused_stats = torch.zeros(len(stat_names), device=Z_rep.device)
    T_perm = torch.zeros(n_per, len(stat_names), device=Z_rep.device) if n_per is not None else None
    N2 = len(Z_rep) - N1
    for i, stat_name in enumerate(stat_names):
        K = get_fuse_km(Z_rep, Z_sigma, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD, stat_name)
        if n_per is not None:
            for j in range(n_per):
                perm = torch.randperm(K.size(0), device=K.device)
                K_perm = K[perm][:, perm]
                T_perm[j, i] = fuse_by_name(K_perm.clone(), N1, N2, stat_name)
                
        fused_stats[i] = fuse_by_name(K.clone(), N1, N2, stat_name)

    if n_per is not None:
        return fused_stats, T_perm

    return fused_stats

def cov_from_B(B, P_rep, P_sigma, N1, N_ip, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD, stat_names, n_perturb, std):
    cache_dir = os.path.join(builtins.CLEAN_PATH, "Sigma_matrix")
    os.makedirs(cache_dir, exist_ok=True)
    stat_names_str = "_".join(sorted(stat_names))
    cache_file = os.path.join(cache_dir, f"{B}_{stat_names_str}_{n_perturb}_{np.round(std, 4)}")
    # b_MMD = 2*torch.median(Pdist(P_rep[:100], P_rep[:100], 2))**2
    # b_VD = b_MMD
    # b_CD = 2*torch.median(Pdist(P_sigma[:100], P_sigma[:100], 2))**2
    # print(b_MMD, b_VD, b_CD)

    param_str = "param"
    if "mean" in stat_names:
        param_str += f"_{np.round(b_MMD, 4)}"
    if "variance" in stat_names:
        param_str += f"_{np.round(b_VD, 4)}"
    if "uncertainty" in stat_names:
        param_str += f"_{np.round(b_CD, 4)}"
    cache_file = cache_file + param_str + ".pt"
    
    # Check if cached file exists
    if os.path.exists(cache_file):
        log(f"Loading cov_B from {cache_file}")
        return torch.load(cache_file, map_location=P_rep.device, weights_only=True)
    
    log(f"Computing cov_B for B={B}, stat_names={stat_names}")

    
    # Precompute kernel matrices for all data to avoid recomputation
    log("Precomputing kernel matrices...")
    precomputed_kernels = {}
    for stat_name in stat_names:
        if stat_name == "mean":
            precomputed_kernels[stat_name] = MD_kernel_matrix(P_rep, b_MMD, kernel_MMD)
            print("MD_kernel_matrix computed")
        elif stat_name == "variance":
            precomputed_kernels[stat_name] = VD_kernel_matrix(P_rep, b_VD, kernel_VD)
            print("VD_kernel_matrix computed")
        elif stat_name == "uncertainty":
            precomputed_kernels[stat_name] = CD_kernel_matrix(P_sigma, b_CD, kernel_CD)
            print("CD_kernel_matrix computed")

    T_b = []
    for i in range(B):
        tot_idx = np.random.choice(len(P_rep), N1+N_ip, replace=False)
        X_idx, Y_idx = tot_idx[:N1], tot_idx[N1:]
        
        fused_stats = torch.zeros(len(stat_names), device=P_rep.device)
        for j, stat_name in enumerate(stat_names):
            all_idx = np.concatenate([X_idx, Y_idx])
            K_sub = precomputed_kernels[stat_name][all_idx][:, all_idx]
            fused_stats[j] = fuse_by_name(K_sub.clone(), N1, N_ip, stat_name)
        
        T_b.append(fused_stats)
        
        # Progress bar every 10 iterations
        # if (i + 1) % 10 == 0 or (i + 1) == B:

        log(f"Progress: {i + 1}/{B} iterations completed", sep=False)
    T_B = torch.stack(T_b, dim=0)
    
    cov_B = torch.cov(T_B.t())

    if cov_B.ndim == 0:
        cov_B = cov_B * torch.eye(len(stat_names)).to(T_B.device)

    # Save to cache file
    log(f"Saving cov_B to {cache_file}")
    torch.save(cov_B.cpu(), cache_file)

    return cov_B
    
def permutation_test(Z_rep, Z_sigma, sigma_B, n, alpha, n_per, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD, stat_names):
    if stat_names is None:
        raise ValueError("stat_names cannot be None")

    fused_stats, T_perm = get_fuse_stats(Z_rep, Z_sigma, n, stat_names, n_per, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD)

    obs_T_agg = fused_stats @ torch.inverse(sigma_B) @ fused_stats.t()

    perm_T_agg = torch.diag(T_perm @ torch.inverse(sigma_B) @ T_perm.t())

    p_value = torch.sum(perm_T_agg > obs_T_agg) / n_per

    # 计数：g = #{perm > obs}, e = #{perm == obs}
    # left  = torch.searchsorted(perm_T_agg, obs_T_agg, right=False)    # #{< obs}
    # right = torch.searchsorted(perm_T_agg, obs_T_agg, right=True)     # #{≤ obs}
    # g = (n_per - right).item()
    # e = (right - left).item()

    # # 随机化 p 值（exact）
    # u = torch.rand(()).item()                                                         # U(0,1)
    # p_value = (g + u * e + 1.0) / (n_per + 1.0)

    # # 精确随机化决策（exact level alpha）
    # target = alpha * (n_per + 1.0)
    # if g > target:
    #     h = False
    # elif g + e < target:
    #     h = True
    # else:
    #     prob = 0.0 if e == 0 else max(0.0, min(1.0, (target - g) / e))                # tie 概率
    #     h = (torch.rand(()).item() < prob)

    h = (p_value < alpha)
    
    return h, float(p_value)