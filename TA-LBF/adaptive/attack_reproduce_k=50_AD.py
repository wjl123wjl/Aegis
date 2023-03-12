
from statistics import mode
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn as nn
from bitstring import Bits
import numpy as np
import torch.nn.functional as F
import os
import copy
import time
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='TA-LBF (targeted attack with limited bit-flips)')

# parser.add_argument('--gpu-id', '-gpu-id', default="7", type=str)


parser.add_argument('--init-lam', '-init_lam', default=2.5, type=float)
parser.add_argument('--init-k', '-init_k', default=50, type=float)
parser.add_argument('--n-aux', '-n_aux', default=128, type=int)


parser.add_argument('--margin', '-margin', default=10, type=float)
parser.add_argument('--max-search-k', '-max_search_k', default=2, type=int)
parser.add_argument('--max-search-lam', '-max_search_lam', default=2, type=int)
parser.add_argument('--ext-max-iters', '-ext_max_iters', default=2000, type=int)
parser.add_argument('--inn-max-iters', '-inn_max_iters', default=5, type=int)
parser.add_argument('--initial-rho1', '-initial_rho1', default=0.0001, type=float)
parser.add_argument('--initial-rho2', '-initial_rho2', default=0.0001, type=float)
parser.add_argument('--initial-rho3', '-initial_rho3', default=0.00001, type=float)
parser.add_argument('--max-rho1', '-max_rho1', default=50, type=float)
parser.add_argument('--max-rho2', '-max_rho2', default=50, type=float)
parser.add_argument('--max-rho3', '-max_rho3', default=5, type=float)
parser.add_argument('--rho-fact', '-rho_fact', default=1.01, type=float)
parser.add_argument('--inn-lr', '-inn_lr', default=0.001, type=float)
parser.add_argument('--stop-threshold', '-stop_threshold', default=1e-4, type=float)
parser.add_argument('--projection-lp', '-projection_lp', default=2, type=int)
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_quan', help='model architecture ')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name')
parser.add_argument('--dataset_name', default='cifar10', type=str, help='dataset, default:cifar10')


class AugLag(nn.Module):
    def __init__(self, n_bits, w, b, step_size, init=False):
        super(AugLag, self).__init__()

        self.n_bits = n_bits
        self.b = nn.Parameter(torch.tensor(b).float(), requires_grad=True)

        self.w_twos = nn.Parameter(torch.zeros([w.shape[0], w.shape[1], self.n_bits]), requires_grad=True)
        self.step_size = step_size
        self.w = w

        base = [2**i for i in range(self.n_bits-1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

        if init:
            self.reset_w_twos()

    def forward(self, x):

        # covert w_twos to float
        w = self.w_twos * self.base
        w = torch.sum(w, dim=2) * self.step_size

        # calculate output
        x = F.linear(x, w, self.b)

        return x

    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] += \
                    torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]), length=self.n_bits).bin])


def project_box(x):
    xp = x
    xp[x>1]=1
    xp[x<0]=0

    return xp


def project_shifted_Lp_ball(x, p):
    shift_vec = 1/2*np.ones(x.size)
    shift_x = x-shift_vec
    normp_shift = np.linalg.norm(shift_x, p)
    n = x.size
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + shift_vec

    return xp


def project_positive(x):
    xp = np.clip(x, 0, None)
    return xp


def loss_func(output, labels, s, t, lam, w, target_thr, source_thr,
              b_ori, k_bits, y1, y2, y3, z1, z2, z3, k, rho1, rho2, rho3):

    l1_1 = torch.max(output[-1][s] - source_thr, torch.tensor(0.0).to(device))
    l1_2 = torch.max(target_thr - output[-1][t], torch.tensor(0.0).to(device))
    l1 = l1_1 + l1_2
    # l1 = output[-1][s] - output[-1][t]
    # l1 = torch.max(output[-1][s] - source_thr, 0)[0] + torch.max(target_thr - output[-1][t], 0)[0]

    # print(target_thr, output[-1][t])
    # print(source_thr, output[-1][s])
    l2 = F.cross_entropy(output[:-1], labels[:-1])
    # print(l1.item(), l2.item())

    y1, y2, y3, z1, z2, z3 = torch.tensor(y1).float().to(device), torch.tensor(y2).float().to(device), torch.tensor(y3).float().to(device), \
                             torch.tensor(z1).float().to(device), torch.tensor(z2).float().to(device), torch.tensor(z3).float().to(device)

    b_ori = torch.tensor(b_ori).float().to(device)
    b = torch.cat((w[s].view(-1), w[t].view(-1)))

    l3 = z1@(b-y1) + z2@(b-y2) + z3*(torch.norm(b - b_ori) ** 2 - k + y3)

    l4 = (rho1/2) * torch.norm(b - y1) ** 2 + (rho2/2) * torch.norm(b - y2) ** 2 \
         + (rho3/2) * (torch.norm(b - b_ori)**2 - k_bits + y3) ** 2

    return l1 + lam * l2 + l3 + l4


def attack(auglag_ori, all_data, labels, labels_cuda, clean_output,
           attack_idx, target_class, source_class, aux_idx,
           lam, k, args):
    # set parameters
    n_aux = args.n_aux
    lam = lam
    ext_max_iters = args.ext_max_iters
    inn_max_iters = args.inn_max_iters
    initial_rho1 = args.initial_rho1
    initial_rho2 = args.initial_rho2
    initial_rho3 = args.initial_rho3
    max_rho1 = args.max_rho1
    max_rho2 = args.max_rho2
    max_rho3 = args.max_rho3
    rho_fact = args.rho_fact
    k_bits = k
    inn_lr = args.inn_lr
    margin = args.margin
    stop_threshold = args.stop_threshold

    projection_lp = args.projection_lp

    all_idx = np.append(aux_idx, attack_idx)

    sub_max = clean_output[attack_idx][[i for i in range(len(clean_output[-1])) if i != source_class]].max()
    target_thr = sub_max + margin
    source_thr = sub_max - margin

    auglag = copy.deepcopy(auglag_ori)

    b_ori_s = auglag.w_twos.data[source_class].view(-1).detach().cpu().numpy()
    b_ori_t = auglag.w_twos.data[target_class].view(-1).detach().cpu().numpy()
    b_ori = np.append(b_ori_s, b_ori_t)
    b_new = b_ori

    y1 = b_ori
    y2 = y1
    y3 = 0

    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y1)
    z3 = 0

    rho1 = initial_rho1
    rho2 = initial_rho2
    rho3 = initial_rho3

    stop_flag = False
    # start_time = time.time()
    for ext_iter in range(ext_max_iters):

        y1 = project_box(b_new + z1 / rho1)
        y2 = project_shifted_Lp_ball(b_new + z2 / rho2, projection_lp)
        y3 = project_positive(-np.linalg.norm(b_new - b_ori, ord=2) ** 2 + k_bits - z3 / rho3)

        for inn_iter in range(inn_max_iters):

            input_var = torch.autograd.Variable(all_data[all_idx], volatile=True)
            target_var = torch.autograd.Variable(labels_cuda[all_idx].long(), volatile=True)

            output = auglag(input_var)
            loss = loss_func(output, target_var, source_class, target_class, lam, auglag.w_twos,
                             target_thr, source_thr,
                             b_ori, k_bits, y1, y2, y3, z1, z2, z3, k_bits, rho1, rho2, rho3)

            loss.backward(retain_graph=True)
            auglag.w_twos.data[target_class] = auglag.w_twos.data[target_class] - \
                                               inn_lr * auglag.w_twos.grad.data[target_class]
            auglag.w_twos.data[source_class] = auglag.w_twos.data[source_class] - \
                                               inn_lr * auglag.w_twos.grad.data[source_class]
            auglag.w_twos.grad.zero_()
       
        # print(inn_iter)
        b_new_s = auglag.w_twos.data[source_class].view(-1).detach().cpu().numpy()
        b_new_t = auglag.w_twos.data[target_class].view(-1).detach().cpu().numpy()
        b_new = np.append(b_new_s, b_new_t)

        if True in np.isnan(b_new):
            return -1
        # if time.time() - start_time >1800:
        #     print("Time Out!!!")
        #     return -1
        z1 = z1 + rho1 * (b_new - y1)
        z2 = z2 + rho2 * (b_new - y2)
        z3 = z3 + rho3 * (np.linalg.norm(b_new - b_ori, ord=2) ** 2 - k_bits + y3)

        rho1 = min(rho_fact * rho1, max_rho1)
        rho2 = min(rho_fact * rho2, max_rho2)
        rho3 = min(rho_fact * rho3, max_rho3)

        temp1 = (np.linalg.norm(b_new - y1)) / max(np.linalg.norm(b_new), 2.2204e-16)
        temp2 = (np.linalg.norm(b_new - y2)) / max(np.linalg.norm(b_new), 2.2204e-16)

        if ext_iter % 50 == 0:
            print('iter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, max(temp1, temp2), loss.item()))
        
        if max(temp1, temp2) <= stop_threshold and ext_iter > 100:
            print('END iter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, max(temp1, temp2), loss.item()))
            stop_flag = True
            break

    auglag.w_twos.data[auglag.w_twos.data > 0.5] = 1.0
    auglag.w_twos.data[auglag.w_twos.data < 0.5] = 0.0
    
    # print(source_w_twos.eq(auglag.w_twos.data[source_class]))
    # print(target_w_twos.eq(auglag.w_twos.data[target_class]))
    print(np.where((auglag_ori.w_twos.data[source_class].eq(auglag.w_twos.data[source_class])).cpu().numpy()==False))
    # print(target_w_twos.eq(auglag.w_twos.data[target_class]))
    print(np.where((auglag_ori.w_twos.data[target_class].eq(auglag.w_twos.data[target_class])).cpu().numpy()==False))
    
    
    output = auglag(all_data)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze(1)
    pa_acc = len([i for i in range(len(output)) if labels_cuda[i] == pred[i] and i != attack_idx and i not in aux_idx]) / \
                     (len(labels) - 1 - n_aux)

    n_bit = torch.norm(auglag_ori.w_twos.data.view(-1) - auglag.w_twos.data.view(-1), p=0).item()

    ret = {
        "pa_acc": pa_acc,
        "stop": stop_flag,
        "suc": target_class == pred[attack_idx].item(),
        "n_bit": n_bit,
        "auglag":auglag,
    }
    print("PA_ACC:", pa_acc*100, " N_flip:", n_bit, " ORIG:", source_class, " PRED:", pred[attack_idx].item())
    return ret


def Update_FC_Weight_for_vgg16(model, auglag, index, num_branch):
    w = auglag.w_twos * auglag.base
    w = torch.sum(w, dim=2) * auglag.step_size
    model_dict = model.state_dict()

    if index == num_branch-1:
        dict_name = ['1.classifier.3.weight']
    else:
        dict_name = ['1.features.'+str(index)+'.output.quan_layer_branch.weight', '1.features.'+str(index)+'.output.linear.weight', '1.features.'+str(index)+'.output.weight']
    if  dict_name[0] in list(model_dict.keys()):
        print("Choose-Dict-Option-1")
        model_dict[dict_name[0]] = w
        model.load_state_dict(model_dict)
    elif dict_name[1] in list(model_dict.keys()):
        print("Choose-Dict-Option-2")
        model_dict[dict_name[1]] = w
        model.load_state_dict(model_dict)
    else:
        print("Choose-Dict-Option-3")
        model_dict[dict_name[2]] = w
        model.load_state_dict(model_dict)
    return 

def Update_FC_Weight_for_resnet(model, auglag, index, num_branch):
    w = auglag.w_twos * auglag.base
    w = torch.sum(w, dim=2) * auglag.step_size
    model_dict = model.state_dict()

    if index == num_branch-1:
        dict_name = ['1.classifier.weight']
    else:
        dict_name = ['1.stage_1.'+str(index)+'.output.quan_layer_branch.weight', '1.stage_1.'+str(index)+'.output.linear.weight', 
                    '1.stage_2.'+str(index-5)+'.output.quan_layer_branch.weight', '1.stage_2.'+str(index-5)+'.output.linear.weight', 
                    '1.stage_3.'+str(index-10)+'.output.quan_layer_branch.weight', '1.stage_3.'+str(index-10)+'.output.linear.weight']
    if  dict_name[0] in list(model_dict.keys()):
        print("Choose-Dict-Option-1")
        model_dict[dict_name[0]] = w
        model.load_state_dict(model_dict)
    elif dict_name[1] in list(model_dict.keys()):
        print("Choose-Dict-Option-2")
        model_dict[dict_name[1]] = w
        model.load_state_dict(model_dict)
    elif dict_name[2] in list(model_dict.keys()):
        print("Choose-Dict-Option-3")
        model_dict[dict_name[2]] = w
        model.load_state_dict(model_dict)
    elif dict_name[3] in list(model_dict.keys()):
        print("Choose-Dict-Option-4")
        model_dict[dict_name[3]] = w
        model.load_state_dict(model_dict)
    elif dict_name[4] in list(model_dict.keys()):
        print("Choose-Dict-Option-5")
        model_dict[dict_name[4]] = w
        model.load_state_dict(model_dict)
    else:
        print("Choose-Dict-Option-6")
        model_dict[dict_name[5]] = w
        model.load_state_dict(model_dict)
    return 

def main():
    np.random.seed(512)

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    
    print(args)

    # prepare the data
    print("Prepare data ... ")
    arch = args.arch
    dataset_name = args.dataset_name
    network_name = args.model_name
    bit_length = 8
    # Number = 0
    weight_list, bias_list, step_size_list = load_model(arch+'1', network_name, dataset_name, device)
    
    # load dataset
    if not dataset_name == 'stl10' and not dataset_name == 'tinyimagenet':
        bs = 256
    else:
        bs = 128
    dataset = load_dataset(dataset_name, batch_size=bs, doNormalization=False)
    val_loader = dataset.test_loader  

    # load the index of attacked samples and the target class of them
    attack_info = np.loadtxt(config.info_root[dataset_name]).astype(int)
    # useful_linear for weight_list
    useful_linear = config.useful_linear[network_name]
    # load num_branch
    num_branch = config.num_branch[network_name]
    # escape num_branch for our validate 
    escape_num = config.escape_num[dataset_name][network_name]
    # confidence threshole for our validate
    conf_th = config.confidence_threshold[dataset_name][network_name]
    mask_num = config.mask_num[dataset_name][network_name]
    is_modify = config.is_modify[dataset_name][network_name]

    model = load_model_normal(arch, network_name, dataset_name, doNorLayer=True, device=device)
    if is_modify:
        index_list = validate_modify(val_loader, model, num_branch, device)
        print("Modify the branches!!", index_list)
    else:
        index_list = [i for i in range(num_branch)]

    # do valiation for our before attack
    our_ori_acc_top1,_,_,our_ori_exit_list = validate_for_attack(val_loader, model, num_branch, index_list, escape_num, mask_num, conf_th, device = device)
    print("Validation Length: ", len(our_ori_acc_top1))
    our_acc_ori = np.mean(our_ori_acc_top1)
    
    print("\n[OUR-ORI] ACC(TOP-1):{0:.4f}\n".format(our_acc_ori))

    index_list_escape = [index_list[i] for i in range(len(index_list)) if i>=escape_num]
    print("Index list after escaping:", index_list_escape, " --Length: ", len(index_list_escape))
    # num=0 # 3 branches
    # index_branch =  config.choose_branch[dataset_name][network_name][num]
    # print("Choose %d branches' index:"%len(index_branch), index_branch)
    # original results for TA-LBF before using our arch.
    # ori_asr = []
    # ori_pa_acc = []
    # ori_n_bit = []
    # ori_n_stop = []
    # ori_param_lam = []
    # ori_param_k_bits = []
    # results for TA-LBF after using our arch.
    asr = []
    pa_acc=[]
    n_bit = []
    n_bit_all = []
    print("Attack Start")
    time0 = time.time()
    for i, (target_class, attack_idx) in enumerate(attack_info):

        # original results for TA-LBF before using our arch.
        ori_asr_1 = []
        ori_pa_acc_1 = []
        ori_n_bit_1 = []
        ori_n_stop_1 = []
        ori_param_lam_1 = []
        ori_param_k_bits_1 = []
        n_bit_all_1 = []

        weight = [weight_list[i] for i in useful_linear]
        bias = [bias_list[i] for i in useful_linear]
        step_size = [step_size_list[i] for i in useful_linear]
        model = load_model_normal(arch, network_name, dataset_name, doNorLayer= True, device=device)
        # for each index, attack once
        for index in index_list_escape:
            dim = weight[index].shape[-1]
            print("[{0}]".format(i)," [Index%s] Dimension of weight: "%str(index), dim)
            all_data, labels= load_data(arch+'1', network_name, dataset_name, device, dim, index, dataset) #add two parameters: index and dataset, minus an output: dataet --- output[index]
            labels_cuda = labels.to(device)
            auglag = AugLag(bit_length, weight[index], bias[index], step_size[index], init=True).to(device)
            clean_output = auglag(all_data)
            source_class = int(labels[attack_idx])
            aux_idx = np.random.choice([i for i in range(len(labels)) if i != attack_idx], args.n_aux, replace=False)

            suc = False
            time_out = False
            flag =False
            cur_k = args.init_k/len(index_list_escape)
            # for search_k in range(args.max_search_k):
            print("k", cur_k)
            cur_lam = args.init_lam
            for search_lam in range(args.max_search_lam):
                print("lam", cur_lam)
                res = attack(auglag, all_data, labels, labels_cuda, clean_output,
                                attack_idx, target_class, source_class, aux_idx,
                                cur_lam, cur_k, args)

                if res == -1:
                    print("Error[{0}]: Lambda:{1} K_bits:{2}".format(i, cur_lam, cur_k))
                    res = {"auglag":auglag}
                    cur_lam = cur_lam / 2.0
                    # time_out = True
                    if search_lam == args.max_search_lam-1:
                        flag = True
                    # break
                    continue
                elif res["suc"]:
                    print("SUCCESS!!")
                    ori_n_stop_1.append(int(res["stop"]))
                    ori_asr_1.append(int(res["suc"]))
                    ori_pa_acc_1.append(res["pa_acc"])
                    ori_n_bit_1.append(res["n_bit"])
                    n_bit_all_1.append(res["n_bit"])
                    ori_param_lam_1.append(cur_lam)
                    ori_param_k_bits_1.append(cur_k)
                    suc = True
                    break

                cur_lam = cur_lam / 2.0

                # if suc or time_out:
                #     break

                # cur_k = cur_k * 2.0       
        

            if not suc and not flag:
                ori_asr_1.append(0)
                ori_n_stop_1.append(0)
                n_bit_all_1.append(res["n_bit"])
                print("[ORI] [{0}] Fail! N_flip:{1}".format(i, n_bit_all_1[-1]))
            else:
                print("[ORI] [{0}] PA-ACC:{1:.4f} Success:{2} N_flip:{3} Stop:{4} Lambda:{5} K:{6}".format(
                    i, ori_pa_acc_1[-1]*100, bool(ori_asr_1[-1]), ori_n_bit_1[-1], bool(ori_n_stop_1[-1]), ori_param_lam_1[-1],
                    ori_param_k_bits_1[-1]))
            
            # modify the model
            if network_name == 'vgg16':
                Update_FC_Weight_for_vgg16(model, res["auglag"], index, num_branch)
            if network_name == 'resnet32':
                Update_FC_Weight_for_resnet(model, res["auglag"], index, num_branch)    
            
            
        print("\n[OUR-ORI] Attack Exit:{0}".format(our_ori_exit_list[attack_idx])) # print original exit entrance index for the attacked sample    
        # do valiation for our after attack
        our_acc_top1,_,our_pred,our_exit_list = validate_for_attack(val_loader, model, num_branch, index_list, escape_num, mask_num, conf_th, device = device)
        ## only for the attacked sample
        ### if exit early : asr.append(0) ; elif exit from the last entrance : asr.append( int(target_class == pred[attack_idx].item()) , n_bit.append(res["n_bit"])) 
        print("\n[OUR-AD] Attack Exit:{0}".format(our_exit_list[attack_idx]))
        
        asr.append(int(target_class == our_pred[attack_idx].item()))
        if asr[-1] == 1:
            n_bit.append(np.sum(ori_n_bit_1))
            print("SUCCESS!! Final! [{0}] N_flip:{1} \n\n".format(i, n_bit[-1]))
        else:
            print("Fail!! Final!!")

        n_bit_all.append(np.sum(n_bit_all_1))
        print("[{0}] N_flip_all:{1:.4f}".format(i, n_bit_all[-1]))
        ## for other samples(except for the attacked sample and auxiliary samples, How to use val_loader?) pa_acc.append(acc)
        sum = 0
        for j in range(len(labels)):
            if j !=attack_idx and j not in aux_idx:
                sum += our_acc_top1[j]

        pa_acc.append(sum/(len(labels)-1-args.n_aux))
        print("[OUR-AD] acc for %s:"%str(i), pa_acc)
        
        if (i+1) % 5 == 0:
            # print("[ORI] END[{0}] PA-ACC:{1:.4f} ASR:{2} N_flip:{3:.4f}".format(
            #     i, np.mean(ori_pa_acc)*100, np.mean(ori_asr)*100, np.mean(ori_n_bit)))
            print("[OUR-AD] END[{0}] PA-ACC:{1:.4f} ASR:{2} N_flip:{3:.4f} N_flip_len:{4} N_flip_all:{5:.4f} N_flip_all_len:{6}".format(
                i, np.mean(pa_acc), np.mean(asr)*100, np.mean(n_bit), len(n_bit), np.mean(n_bit_all), len(n_bit_all)))
        
    print("\n\n")
    # print("[ORI] END Original_ACC:{0:.4f} PA_ACC:{1:.4f} ASR:{2:.2f} N_flip:{3:.4f}".format(
    #         0*100, np.mean(ori_pa_acc)*100, np.mean(ori_asr)*100, np.mean(ori_n_bit)))
    print("[OUR-AD] END Original_ACC:{0:.4f} PA_ACC:{1:.4f} ASR:{2:.2f} N_flip:{3:.4f} N_flip_len:{4} N_flip_all:{5:.4f} N_flip_all_len:{6}".format(
            our_acc_ori, np.mean(pa_acc), np.mean(asr)*100, np.mean(n_bit), len(n_bit), np.mean(n_bit_all), len(n_bit_all)))
    
    print("total time: ",time.time()-time0)

if __name__ == '__main__':
    main()
