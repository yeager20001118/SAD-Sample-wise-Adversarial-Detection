import torchvision.transforms as transforms
from torchvision import datasets
import torch
import numpy as np
import builtins


# UDF authored by Alex, Jiacheng, Xunye, Yiyi, Zesheng, and Zhijian
builtins.IS_LOG = True
builtins.DATALOG_COUNT = 0
def log(*args, sep=False, **kwargs):
    if builtins.IS_LOG:
        if sep:
            msg = " ".join(map(str, args))
            total_len = 80
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

def load_data(path, N, rs, check, model, ref, is_test=True):
    if 'cifar10' in path:
        samples = sample_CIFAR10(path, N, rs, check, model, ref, is_test)
    return samples

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def sample_CIFAR10(path, N, rs, check, model, ref="adv", is_test=True, batch_size=128):
    """ Default reference set is adv """
    device = check_device()
    model = model.to(device)
    if path.endswith(".npz"):
        ADV, n_total = load_cifar10_adv_success(path)
        
        if not is_test and builtins.DATALOG_COUNT == 0:
            log(f"Loaded {n_total} total samples, {len(ADV)} successfully attacked samples", sep=True)
    else:
        ADV = np.load(path)
    ADV = torch.from_numpy(ADV).float().to(device)
    ADV_rep = rep_by_batch(ADV, model, batch_size)

    # shuffle
    np.random.seed(10086+rs)
    ind = np.random.choice(len(ADV), len(ADV), replace=False)
    ADV, ADV_rep = ADV[ind], ADV_rep[ind]
    
    if ref == "adv":
        n_samples = len(ADV)
        split_idx = n_samples // 2
        if not is_test and builtins.DATALOG_COUNT == 0:
            log("ADV as ref", sep=True)
            log("ADV.shape: {}".format(ADV.shape))
        if check:
            P = ADV[:split_idx]
            Q = ADV[split_idx:]
            P_rep = ADV_rep[:split_idx] 
            Q_rep = ADV_rep[split_idx:]
        else:
            P = ADV #[:split_idx]
            Q = load_cifar10_test().to(device) #[:split_idx].to(device) # Q is shuffled inside
            P_rep = ADV_rep #[:split_idx]
            Q_rep = rep_by_batch(Q, model, batch_size)

    np.random.seed(rs*N)
    P_tr_idx = np.random.choice(len(P), N, replace=False)
    np.random.seed((rs+918)*N)
    Q_tr_idx = np.random.choice(len(Q), N, replace=False)
    
    P_mask = torch.zeros(len(P), dtype=torch.bool)
    Q_mask = torch.zeros(len(Q), dtype=torch.bool)
    P_mask[P_tr_idx] = True
    Q_mask[Q_tr_idx] = True
    
    P_tr, P_te = P[P_mask], P[~P_mask]
    Q_tr, Q_te = Q[Q_mask], Q[~Q_mask]
    P_rep_tr, P_rep_te = P_rep[P_mask], P_rep[~P_mask]
    Q_rep_tr, Q_rep_te = Q_rep[Q_mask], Q_rep[~Q_mask]
    
    del P, Q, P_rep, Q_rep
    del P_mask, Q_mask, P_tr_idx, Q_tr_idx

    if is_test:
        if builtins.DATALOG_COUNT == 0:
            log("Testing data all")
            log("P.shape: {}, Q.shape: {}, P_rep.shape: {}, Q_rep.shape: {}".format(P_te.shape, Q_te.shape, P_rep_te.shape, Q_rep_te.shape))
            log("end loading test data", sep=True)
            builtins.DATALOG_COUNT += 1
        return (flatten_img(P_te), flatten_img(Q_te)), (P_rep_te, Q_rep_te)
    else:
        if builtins.DATALOG_COUNT == 0:
            log("Training data")
            log("P.shape: {}, Q.shape: {}, P_rep.shape: {}, Q_rep.shape: {}".format(P_tr.shape, Q_tr.shape, P_rep_tr.shape, Q_rep_tr.shape))
            log("end loading train data", sep=True)
        return (flatten_img(P_tr), flatten_img(Q_tr)), (P_rep_tr, Q_rep_tr)
        

def rep_by_batch(data, model,batch_size):
    n = data.shape[0]
    n_batches = (n - 1) // batch_size + 1
    outputs = []

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n)
            batch = data[start_idx:end_idx]
            batch_output = model(batch)
            outputs.append(batch_output)

    return torch.cat(outputs, dim=0)

def flatten_img(img):
    return img.view(img.size(0), -1)

def load_cifar10_test():
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = datasets.CIFAR10(root='/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection/data/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=0) # shuffle Q
    imgs, _ = next(iter(test_loader))
    return imgs

def load_cifar10_adv_success(path):
    data = np.load(path)
    X_adv = data['X_adv']
    original_labels = data['predicted_original_labels']
    predicted_labels = data['predicted_adv_labels']
    
    success_mask = (predicted_labels != original_labels)
    ADV = X_adv[success_mask]
    return ADV, len(X_adv)