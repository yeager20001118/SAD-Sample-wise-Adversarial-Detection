import numpy as np
import argparse
import sys
import os
import builtins

# UDF authored by Alex, Jiacheng, Xunye, Yiyi, Zesheng, and Zhijian
sys.path.append('/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection')
from models import *
from exp.dataloader import *
# from baselines.EPS_AD_xunye.EPS_AD import train_EPS_AD, EPS_AD
# from baselines.SAMMD.SAMMD import train_SAMMD, SAMMD
# from baselines.MMDAgg.MMDAgg import MMDAgg
# from baselines.MMDFUSE.MMD_Fuse import MMDFuse
# from baselines.DUAL.MMD_DUAL import DUAL_no_train
from baselines.SAD.SAD_ZJ import SAD

# builtins.IS_LOG = False  # mute the log
print("Logging is {}".format("on" if builtins.IS_LOG else "off"))

parser = argparse.ArgumentParser()
# parser.add_argument('--attk_method', type=str, default='pgd', help='adversarial attack method')
# parameters to generate data
parser.add_argument('--check', default=0, help='check reject adv (1), reject clean(0)', type=int)
parser.add_argument('--N_rf', default=50, help='number of samples in referenced data', type=int)
parser.add_argument('--N_ip', default=50, help='number of samples in input data', type=int)
parser.add_argument('--epsilon', default=[1,2,4,8], help='epsilon', type=list)
parser.add_argument('--rs', default=[819,819,819,819], help='random seed', type=list)

# parameters to conduct exp
parser.add_argument('--n_exp', default=10, help='number of experiments', type=int)
parser.add_argument('--n_test', default=100, help='number of test times', type=int)
parser.add_argument('--n_per', default=100, help='number of permutations', type=int)
parser.add_argument('--alpha', default=0.05, help='probability of not reject adv', type=float)

# parameters to train SAD
parser.add_argument('--kernel', default="log_rbf", help='kernel type', type=str)
parser.add_argument('--N_epoch', default=0, help='epochs to update kernel params', type=int)
parser.add_argument('--batch_size', default=128, help='batch size', type=int)
parser.add_argument('--lr', default=0.0005, help='learning rate', type=float)
parser.add_argument('--ref', default="org", help='reference data, adv or org', type=str)

# DUAL specific parameters
parser.add_argument('--n_bandwidth', default=[5,5], help='number of bandwidths', type=list)
parser.add_argument('--reg', default=1e-8, help='regularization', type=float)
parser.add_argument('--way', default=['Agg','Fuse'], help='kernel type', type=list)
parser.add_argument('--is_cov', default=False, help='is covariance matrix', type=bool)

# parameters to train EPS-AD
parser.add_argument('--kernel_epsad', default="eps_ad", help='kernel type (kept for compatibility)', type=str)
parser.add_argument('--N_epoch_epsad', default=200, help='epochs to train EPS-AD discriminator', type=int)
parser.add_argument('--batch_size_epsad', default=128, help='batch size', type=int)
parser.add_argument('--lr_epsad', default=0.0002, help='learning rate', type=float)

# EPS-AD specific parameters
parser.add_argument('--feature_dim', default=300, help='feature dimension for discriminator', type=int)
parser.add_argument('--sigma0_init', default=15.0, help='initial sigma0 value', type=float)
parser.add_argument('--sigma_init', default=100.0, help='initial sigma value', type=float)
parser.add_argument('--epsilon_init', default=2, help='initial epsilon value', type=int)

args = parser.parse_args()

Results = np.zeros((8, args.n_exp))

H_EPS_AD = np.zeros(args.n_test)
H_SAMMD = np.zeros(args.n_test)
H_MMDAgg = np.zeros(args.n_test)
H_MMDFUSE = np.zeros(args.n_test)
H_DUAL = np.zeros(args.n_test)
H_SAD_com1 = np.zeros(args.n_test)
H_SAD_com2 = np.zeros(args.n_test)
H_SAD_com3 = np.zeros(args.n_test)

builtins.CLEAN_PATH = "/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection/data/cifar10"
# pre-settings for adv path
adv_path = "/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection/adv"
# pre-settings for model
model_arch = "Res18"
ckpt = "resnet-18.pth"
model_path = os.path.join(adv_path + "/checkpoint/CIFAR10", model_arch, ckpt)
# pre-settings for adv attack data
dataset_name = 'cifar10'
attk_method = 'pgd'
n_steps = 5
penalty = "linf"

model = load_model(model_arch, model_path)

exp_path = "/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection/exp/motivation/"
for i in range(len(args.epsilon)):
    # if i != 1:
    #     continue
    builtins.DATALOG_COUNT = 0
    path = os.path.join(adv_path, "Adv_data", dataset_name, model_arch, "with_labels", f"Adv_{dataset_name}_{attk_method}_{n_steps}_eps{args.epsilon[i]}_{penalty}.npz")

    # params_epsad = train_EPS_AD(path, args.N_rf, args.rs[i], args.check, model, args.N_epoch_epsad, args.lr_epsad, 
    #                      dataset=dataset_name, feature_dim=args.feature_dim, 
    #                      sigma0_init=args.sigma0_init, sigma_init=args.sigma_init, 
    #                      epsilon_init=args.epsilon_init, ref=args.ref)
    # params_sammd = train_SAMMD(path, args.N_rf, args.rs[i], args.check, model, args.N_epoch, args.lr, args.ref)
    # params = train_SAD(path, args.N_rf, args.rs[i], args.check, model, args.N_epoch, args.lr, args.ref)

    file_name = args.ref+'_'+dataset_name+'_'+attk_method+'_'+str(n_steps)+'_eps'+str(args.epsilon[i])+'_'+penalty+'_Nrf'+str(args.N_rf)+'_Nip'+str(args.N_ip)
    setup_time_log()
    for kk in range(args.n_exp):

        # H_EPS_AD, _, _, _ = EPS_AD(path, args.N_rf, args.N_ip, kk*args.n_exp+args.rs[i], args.check, params_epsad, args.kernel_epsad, args.n_test, args.n_per, args.alpha, args.ref, model)

        # H_SAMMD, _, _, _ = SAMMD(path, args.N_rf, args.N_ip, kk*args.n_exp+args.rs[i], args.check, [model] + params_sammd, args.n_test, args.n_per, args.alpha, args.ref)
        
        # H_MMDAgg, _, _, _ = MMDAgg(path, args.N_rf, args.N_ip, kk*args.n_exp+args.rs[i], args.check, model, args.n_test, args.n_per, args.alpha, args.ref)

        # H_MMDFUSE, _, _, _ = MMDFuse(path, args.N_rf, args.N_ip, kk*args.n_exp+args.rs[i], args.check, model, args.n_test, args.n_per, args.alpha, args.ref)

        # H_DUAL, _, _, _ = DUAL_no_train(path, args.N_rf, args.N_ip, kk*args.n_exp+args.rs[i], args.check, model, args.n_test, args.n_per, args.alpha, args.n_bandwidth, args.reg, args.way, args.is_cov, args.ref)
        
        H_SAD_com1, _, _, _ = SAD(path, args.N_rf, args.N_ip, kk*args.n_exp+args.rs[i], args.check, model, "log_rbf", args.n_test, args.n_per, args.alpha, args.ref)

        # H_SAD_com2, _, _, _ = SAD(path, args.N_rf, args.N_ip, kk*args.n_exp+args.rs[i], args.check, model, "airm", args.n_test, args.n_per, args.alpha, args.ref)

        # H_SAD_com3, _, _, _ = SAD(path, args.N_rf, args.N_ip, kk*args.n_exp+args.rs[i], args.check, model, "stein", args.n_test, args.n_per, args.alpha, args.ref)
        # break
        Results[0, kk] = np.mean(H_EPS_AD)
        Results[1, kk] = np.mean(H_SAMMD)
        Results[2, kk] = np.mean(H_MMDAgg)
        Results[3, kk] = np.mean(H_MMDFUSE)
        Results[4, kk] = np.mean(H_DUAL)
        Results[5, kk] = np.mean(H_SAD_com1)
        Results[6, kk] = np.mean(H_SAD_com2)
        Results[7, kk] = np.mean(H_SAD_com3)

        # if args.check == 1:
        #     os.makedirs(os.path.join(exp_path, "Results", "test_power", str(args.alpha)), exist_ok=True)
        #     np.savetxt(os.path.join(exp_path, "Results", "test_power", str(args.alpha), file_name), Results, fmt='%.3f')
        # if args.check == 0:
        #     os.makedirs(os.path.join(exp_path, "Results", "typeI_error", str(args.alpha)), exist_ok=True)
        #     np.savetxt(os.path.join(exp_path, "Results", "typeI_error", str(args.alpha), file_name), Results, fmt='%.3f')
        break
    Final_results = np.zeros((Results.shape[0], 2))
    for j in range(Results.shape[0]):
        Final_results[j, 0] = np.mean(Results[j, :])
        Final_results[j, 1] = np.std(Results[j, :])/np.sqrt(args.n_exp)

    # if args.check == 1:
    #     result_file = os.path.join(exp_path, "Results", "test_power", str(args.alpha), file_name)
    #     with open(result_file, 'a') as f:
    #         np.savetxt(f, Final_results, fmt='%.3f')
    #     print(f"Detection of {args.ref}-ref {dataset_name} under {attk_method} with eps={args.epsilon[i]}, penalty={penalty} and Nrf={args.N_rf} and Nip={args.N_ip} is done")
    # if args.check == 0:
    #     result_file = os.path.join(exp_path, "Results", "typeI_error", str(args.alpha), file_name)
    #     with open(result_file, 'a') as f:
    #         np.savetxt(f, Final_results, fmt='%.3f')
    #     print(f"TypeI check of {args.ref}-ref {dataset_name} under {attk_method} with eps={args.epsilon[i]}, penalty={penalty} and Nrf={args.N_rf} and Nip={args.N_ip} is done")

    print("EPS-AD: {:.3f} ± {:.3f}".format(Final_results[0, 0], Final_results[0, 1]))
    print("SAMMD: {:.3f} ± {:.3f}".format(Final_results[1, 0], Final_results[1, 1]))
    print("MMDAgg: {:.3f} ± {:.3f}".format(Final_results[2, 0], Final_results[2, 1]))
    print("MMDFUSE: {:.3f} ± {:.3f}".format(Final_results[3, 0], Final_results[3, 1]))
    print("DUAL: {:.3f} ± {:.3f}".format(Final_results[4, 0], Final_results[4, 1]))
    print("SAD-com1: {:.3f} ± {:.3f}".format(Final_results[5, 0], Final_results[5, 1]))
    print("SAD-com2: {:.3f} ± {:.3f}".format(Final_results[6, 0], Final_results[6, 1]))
    print("SAD-com3: {:.3f} ± {:.3f}".format(Final_results[7, 0], Final_results[7, 1]))
    break