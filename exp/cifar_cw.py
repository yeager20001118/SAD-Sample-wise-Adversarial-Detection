import numpy as np
import argparse
import sys
import os
import builtins

# UDF authored by Alex, Jiacheng, Xunye, Yiyi, Zesheng, and Zhijian
sys.path.append('/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection')
from models import *
from exp.dataloader import *
from baselines.SAD.SAD import train_SAD, SAD

# builtins.IS_LOG = False  # mute the log
print("Logging is {}".format("on" if builtins.IS_LOG else "off"))

parser = argparse.ArgumentParser()
# parser.add_argument('--attk_method', type=str, default='pgd', help='adversarial attack method')
# parameters to generate data
parser.add_argument('--check', default=1, help='check reject adv (1), reject clean(0)')
parser.add_argument('--N1', default=5, help='number of samples in P')
parser.add_argument('--epsilon', default=[1,2,4,8], help='epsilon')
parser.add_argument('--rs', default=[819,819,819,819], help='random seed')

# parameters to conduct exp
parser.add_argument('--n_exp', default=10, help='number of experiments')
parser.add_argument('--n_test', default=100, help='number of test times')
parser.add_argument('--n_per', default=1000, help='number of permutations')
parser.add_argument('--alpha', default=0.05, help='probability of not reject adv')

# parameters to train SAD
parser.add_argument('--kernel', default="com3", help='kernel type')
parser.add_argument('--N_epoch', default=0, help='epochs to update kernel params')
parser.add_argument('--batch_size', default=128, help='batch size')
parser.add_argument('--lr', default=0.0005, help='learning rate')
parser.add_argument('--ref', default="adv", help='reference data, adv or org')

args = parser.parse_args()

Results = np.zeros((3, args.n_exp))

H_SAD_com1 = np.zeros(args.n_test)
H_SAD_com2 = np.zeros(args.n_test)
H_SAD_com3 = np.zeros(args.n_test)

# pre-settings for adv path
adv_path = "/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection/adv"
# pre-settings for model
model_arch = "Res18"
ckpt = "net_150.pth"
model_path = os.path.join(adv_path, model_arch + "_ckpt", ckpt)
# pre-settings for adv attack data
dataset_name = 'cifar10'
attk_method = 'cw'
n_steps = 5
penalty = "l2"

model = load_model(model_arch, model_path)

exp_path = "/data/gpfs/projects/punim2112/SAD-Sample-wise-Adversarial-Detection/exp/"
for i in range(len(args.epsilon)):
    builtins.DATALOG_COUNT = 0
    path = os.path.join(adv_path, "Adv_data", dataset_name, model_arch, f"Adv_{dataset_name}_{attk_method}_{n_steps}_eps{args.epsilon[i]}_{penalty}.npy")
    params = train_SAD(path, args.N1, args.rs[i], args.check, model, args.N_epoch, args.lr, args.ref)
    file_name = args.ref+'_'+dataset_name+'_'+attk_method+'_'+str(n_steps)+'_eps'+str(args.epsilon[i])+'_'+penalty+'_N'+str(args.N1)
    for kk in range(args.n_exp):
        H_SAD_com1, _, _, _ = SAD(path, args.N1, kk*args.n_exp+args.rs[i], args.check, [model] + params, "com1", args.n_test, args.n_per, args.alpha, args.ref)

        H_SAD_com2, _, _, _ = SAD(path, args.N1, kk*args.n_exp+args.rs[i], args.check, [model] + params, "com2", args.n_test, args.n_per, args.alpha, args.ref)

        H_SAD_com3, _, _, _ = SAD(path, args.N1, kk*args.n_exp+args.rs[i], args.check, [model] + params, "com3", args.n_test, args.n_per, args.alpha, args.ref)

        Results[0, kk] = np.mean(H_SAD_com1)
        Results[1, kk] = np.mean(H_SAD_com2)
        Results[2, kk] = np.mean(H_SAD_com3)

        if args.check == 1:
            os.makedirs(os.path.join(exp_path, "Results", "test_power", str(args.alpha)), exist_ok=True)
            np.savetxt(os.path.join(exp_path, "Results", "test_power", str(args.alpha), file_name), Results, fmt='%.3f')
        if args.check == 0:
            os.makedirs(os.path.join(exp_path, "Results", "typeI_error", str(args.alpha)), exist_ok=True)
            np.savetxt(os.path.join(exp_path, "Results", "typeI_error", str(args.alpha), file_name), Results, fmt='%.3f')

    Final_results = np.zeros((Results.shape[0], 2))
    for j in range(Results.shape[0]):
        Final_results[j, 0] = np.mean(Results[j, :])
        Final_results[j, 1] = np.std(Results[j, :])/np.sqrt(args.n_exp)

    if args.check == 1:
        result_file = os.path.join(exp_path, "Results", "test_power", str(args.alpha), file_name)
        with open(result_file, 'a') as f:
            np.savetxt(f, Final_results, fmt='%.3f')
        print(f"Detection of {args.ref}-ref {dataset_name} under {attk_method} with eps={args.epsilon[i]}, penalty={penalty} and N={args.N1} is done")
    if args.check == 0:
        result_file = os.path.join(exp_path, "Results", "typeI_error", str(args.alpha), file_name)
        with open(result_file, 'a') as f:
            np.savetxt(f, Final_results, fmt='%.3f')
        print(f"TypeI check of {args.ref}-ref {dataset_name} under {attk_method} with eps={args.epsilon[i]}, penalty={penalty} and N={args.N1} is done")

    print("SAD-com1: {:.3f} ± {:.3f}".format(Final_results[0, 0], Final_results[0, 1]))
    print("SAD-com2: {:.3f} ± {:.3f}".format(Final_results[1, 0], Final_results[1, 1]))
    print("SAD-com3: {:.3f} ± {:.3f}".format(Final_results[2, 0], Final_results[2, 1]))