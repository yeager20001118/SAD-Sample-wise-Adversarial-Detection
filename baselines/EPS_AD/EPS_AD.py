import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

sys.path.append('/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection')
from models import *
from exp.dataloader import load_data
from baselines.EPS_AD.D_net import Discriminator, Discriminator_cifar, Discriminator_cifar2
from baselines.EPS_AD.utils_MMD import MMDu, MMD_batch

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def log(message, *args, sep=False):
    """Simple logging function to match SAD.py style"""
    if sep:
        print("=" * 50)
    print(message, *args)
    if sep:
        print("=" * 50)

def train_EPS_AD(path, N1, rs, check, model, N_epoch, lr, dataset='cifar10', feature_dim=300, 
                 sigma0_init=15.0, sigma_init=100.0, epsilon_init=2, ref=None):
    """
    Train EPS-AD discriminator and parameters for adversarial detection.
    
    Args:
        path: Path to adversarial data directory
        N1: Number of samples from each distribution
        rs: Random seed
        check: Check parameter (from original SAD interface)
        model: Classifier model (unused in EPS-AD but kept for interface compatibility)
        N_epoch: Number of training epochs
        lr: Learning rate
        dataset: Dataset name ('cifar10' or 'imagenet')
        feature_dim: Feature dimension for discriminator
        sigma0_init: Initial value for sigma0 parameter
        sigma_init: Initial value for sigma parameter  
        epsilon_init: Initial value for epsilon parameter
        ref: Reference parameter (from original SAD interface)
    
    Returns:
        Dictionary containing trained discriminator and optimized parameters
    """
    device = check_device()
    np.random.seed(rs)
    torch.manual_seed(rs)

    log("Training EPS-AD on path: ", path)
    
    # Initialize image parameters
    img_size = 32 if dataset == 'cifar10' else 224
    n_channels = 3  # RGB images
    
    # Load data using existing dataloader
    (P, Q), (P_rep, Q_rep) = load_data(path, N1, rs, check, model, ref=ref, is_test=False)
    
    # Move data to device
    P, Q = P.to(device), Q.to(device)
    P_rep, Q_rep = P_rep.to(device), Q_rep.to(device)
    
    # Reshape flattened images back to 3D format for discriminator
    # P, Q are flattened (N, C*H*W), need to reshape to (N, C, H, W)
    if len(P.shape) == 2:  # If flattened
        P = P.view(-1, n_channels, img_size, img_size)
        Q = Q.view(-1, n_channels, img_size, img_size)
    
    # Initialize discriminator network
    if dataset == 'cifar10':
        net = Discriminator_cifar(img_size=img_size, feature_dim=feature_dim)
    else:
        net = Discriminator(img_size=img_size, feature_dim=feature_dim)
    net = net.to(device)
    
    # Initialize learnable parameters
    epsilonOPT = torch.log(torch.from_numpy(np.random.rand(1) * 10 ** (-epsilon_init)).to(device, torch.float))
    epsilonOPT.requires_grad = True
    
    sigmaOPT = torch.from_numpy(np.ones(1) * np.sqrt(2 * img_size * img_size * sigma_init)).to(device, torch.float)
    sigmaOPT.requires_grad = True
    
    sigma0OPT = torch.from_numpy(np.ones(1) * np.sqrt(sigma0_init)).to(device, torch.float)
    sigma0OPT.requires_grad = True
    
    # Setup optimizer
    optimizer = torch.optim.Adam(list(net.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=lr)
    
    # Prepare training data
    # Use original (P,Q) as reference data and representation (P_rep, Q_rep) as feature data
    ref_data = torch.cat([P, Q], dim=0)  # Original images for kernel computation
    
    log(f"Starting training for {N_epoch} epochs...")
    
    for epoch in range(N_epoch):
        net.train()
        optimizer.zero_grad()
        
        # Sample batches for training
        batch_size = min(500, P.shape[0])
        indices = torch.randperm(P.shape[0])[:batch_size]
        
        clean_batch = P[indices]
        adv_batch = Q[indices]
        
        # Combine clean and adversarial samples
        X = torch.cat([clean_batch, adv_batch], dim=0)
        
        # Extract features using discriminator
        _, features = net(X, out_feature=True)
        
        # Convert parameters
        ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0_u = sigma0OPT ** 2
        
        # Compute MMD statistic
        TEMP = MMDu(features, clean_batch.shape[0], X.view(X.shape[0], -1), sigma, sigma0_u, ep)
        mmd_value_temp = -1 * (TEMP[0])
        mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        
        # Backward pass
        STAT_u.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            log(f"Epoch {epoch}, MMD: {mmd_value_temp.item():.6f}, STAT: {STAT_u.item():.6f}")
    
    log(f"Training completed. Final MMD: {mmd_value_temp.item():.6f}, STAT: {STAT_u.item():.6f}", sep=True)
    
    # Return trained parameters
    trained_params = {
        'discriminator': net,
        'epsilonOPT': epsilonOPT,
        'sigmaOPT': sigmaOPT,
        'sigma0OPT': sigma0OPT,
        'epsilon': ep,
        'sigma': sigma,
        'sigma0_u': sigma0_u,
        'feature_dim': feature_dim,
        'img_size': img_size,
        'dataset': dataset
    }
    
    return trained_params

def EPS_AD(path, N1, rs, check, model_params, kernel, n_test, n_per, alpha, ref=None, model=None):
    """
    Test EPS-AD method for adversarial detection.
    
    Args:
        path: Path to test data directory
        N1: Number of samples from each distribution
        rs: Random seed
        check: Check parameter (from original SAD interface)
        model_params: Dictionary containing trained discriminator and parameters
        kernel: Kernel parameter (unused in EPS-AD but kept for interface compatibility)
        n_test: Number of test iterations
        n_per: Number of permutations (unused but kept for interface compatibility)
        alpha: Significance level
        ref: Reference parameter (from original SAD interface)
    
    Returns:
        Tuple of (H_EPS_AD, T_EPS_AD, M_EPS_AD, avg_test_time)
        - H_EPS_AD: Binary decisions (1 = clean, 0 = adversarial)
        - T_EPS_AD: Test statistics
        - M_EPS_AD: MMD distances
        - avg_test_time: Average test time per sample
    """
    device = check_device()
    np.random.seed(rs)
    torch.manual_seed(rs)
    
    # Extract trained parameters
    net = model_params['discriminator'].to(device)
    sigma = model_params['sigma'].to(device)
    sigma0_u = model_params['sigma0_u'].to(device)
    ep = model_params['epsilon'].to(device)
    feature_dim = model_params['feature_dim']
    img_size = model_params['img_size']
    dataset = model_params['dataset']
    n_channels = 3  # RGB images
    
    # Load test data
    (P, Q), (P_rep, Q_rep) = load_data(path, N1, rs, check, model, ref=ref)
    P, Q = P.to(device), Q.to(device)
    P_rep, Q_rep = P_rep.to(device), Q_rep.to(device)
    
    # Reshape flattened images back to 3D format for discriminator
    if len(P.shape) == 2:  # If flattened
        P = P.view(-1, n_channels, img_size, img_size)
        Q = Q.view(-1, n_channels, img_size, img_size)
    
    # Prepare reference data for detection
    SIZE = min(500, P.shape[0])
    ref_indices = torch.randperm(P.shape[0])[:SIZE]
    ref_data = P[ref_indices]
    
    H_EPS_AD = np.zeros(n_test)
    T_EPS_AD = np.zeros(n_test)
    M_EPS_AD = np.zeros(n_test)
    
    net.eval()
    test_time = 0
    
    log(f"Starting EPS-AD testing with {n_test} iterations...")
    
    for k in range(n_test):
        # Sample test data
        test_indices = torch.randperm(Q.shape[0])[:N1]
        test_samples = Q[test_indices]  # Test on adversarial samples
        
        start_time = time.time()
        
        with torch.no_grad():
            # Extract features for reference and test samples
            _, feature_ref = net(ref_data, out_feature=True)
            _, feature_test = net(test_samples, out_feature=True)
            
            # Compute MMD-based detection statistic
            combined_features = torch.cat([feature_ref, feature_test], dim=0)
            combined_data = torch.cat([ref_data, test_samples], dim=0).view(ref_data.shape[0] + test_samples.shape[0], -1)
            
            # Compute batch MMD distances for each test sample
            mmd_distances = MMD_batch(combined_features, feature_ref.shape[0], combined_data, sigma, sigma0_u, ep)
            
            # Use mean MMD distance as test statistic
            test_stat = mmd_distances.mean().cpu().item()
            
            # Simple threshold-based decision (can be refined)
            # Lower MMD distances indicate similarity to clean data
            threshold = 0.1  # This should be determined from training/validation
            decision = 1 if test_stat < threshold else 0  # 1 = clean, 0 = adversarial
            
        test_time += time.time() - start_time
        
        H_EPS_AD[k] = decision
        T_EPS_AD[k] = test_stat
        M_EPS_AD[k] = test_stat  # Using same value for MMD
        
        if k % 10 == 0:
            log(f"Test {k}, Stat: {test_stat:.6f}, Decision: {decision}")
    
    avg_test_time = test_time / n_test
    log(f"EPS-AD testing completed. Average test time: {avg_test_time:.4f}s", sep=True)
    
    return H_EPS_AD, T_EPS_AD, M_EPS_AD, avg_test_time

def EPS_AD_sample_wise(path, N1, rs, check, model_params, sample_data, ref=None):
    """
    Sample-wise EPS-AD detection for individual samples.
    
    Args:
        path: Path to data directory
        N1: Number of reference samples
        rs: Random seed
        check: Check parameter
        model_params: Dictionary containing trained discriminator and parameters
        sample_data: Individual sample(s) to test
        ref: Reference parameter
    
    Returns:
        Detection scores for each sample
    """
    device = check_device()
    np.random.seed(rs)
    torch.manual_seed(rs)
    
    # Extract trained parameters
    net = model_params['discriminator'].to(device)
    sigma = model_params['sigma'].to(device)
    sigma0_u = model_params['sigma0_u'].to(device)
    ep = model_params['epsilon'].to(device)
    img_size = model_params['img_size']
    n_channels = 3  # RGB images
    
    # Load reference data
    (P, Q), (P_rep, Q_rep) = load_data(path, N1, rs, check, None, ref=ref)
    P = P.to(device)
    
    # Reshape flattened images back to 3D format for discriminator
    if len(P.shape) == 2:  # If flattened
        P = P.view(-1, n_channels, img_size, img_size)
    
    # Prepare reference data
    SIZE = min(500, P.shape[0])
    ref_indices = torch.randperm(P.shape[0])[:SIZE]
    ref_data = P[ref_indices]
    
    # Ensure sample_data is a tensor
    if not torch.is_tensor(sample_data):
        sample_data = torch.tensor(sample_data)
    sample_data = sample_data.to(device)
    
    # Handle single sample vs batch
    if len(sample_data.shape) == 3:  # Single sample
        sample_data = sample_data.unsqueeze(0)
    
    net.eval()
    scores = []
    
    with torch.no_grad():
        # Extract features for reference data
        _, feature_ref = net(ref_data, out_feature=True)
        
        # Process each sample
        for i in range(sample_data.shape[0]):
            sample = sample_data[i:i+1]  # Keep batch dimension
            
            # Extract features for test sample
            _, feature_test = net(sample, out_feature=True)
            
            # Compute MMD distance
            combined_features = torch.cat([feature_ref, feature_test], dim=0)
            combined_data = torch.cat([ref_data, sample], dim=0).view(ref_data.shape[0] + 1, -1)
            
            mmd_distance = MMD_batch(combined_features, feature_ref.shape[0], combined_data, sigma, sigma0_u, ep)
            scores.append(mmd_distance.cpu().item())
    
    return np.array(scores)