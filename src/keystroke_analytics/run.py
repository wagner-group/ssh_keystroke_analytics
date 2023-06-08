from .data import *
from .model import *
from .visual import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
import time
import string
import argparse
import os

import matplotlib as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#torch.set_num_threads(1)
def binary_search(l, x):
    low, mid, high = 0, 0, len(l)-1
    while low<high: 
        mid = (high + low) // 2
        if l[mid] <= x:
            low = mid + 1
        else:
            high = mid
 
    return high


def train(model, training_data, batch_size, optimizer, scheduler, criterion, epoch=1, device="cpu", max_batches=1000000, FSNet=False, CNNGRU=False, TypeNet=False):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 2500
    start_time = time.time()
    batch = 0
    total_batch_count = training_data.batch_count(batch_size)
    while True:
        with torch.no_grad():
            if not TypeNet:
                cont, data, labels, masks, users, meta, offsets, src_ips, dst_ips = training_data.get_batch(batch_size=batch_size, device=device)
            else:
                cont, data, labels, positions, users, meta = training_data.typenetbatch(batch_size=batch_size, device=device)


        optimizer.zero_grad()
        if not cont:
            return
        batch += 1

        if batch > max_batches:
            return


        if FSNet:
            # Run FSnet optim
            output, rec = model(data, users)
            #loss = (2*criterion(output, labels)/np.log(2) + rec)/2
            loss = criterion(output, labels)
        elif CNNGRU:
            output = model(data, users)
            loss = criterion(output, labels) 
        elif TypeNet:
            output = model(data, users, positions)
            loss = criterion(output, labels)
        else:
            if users is not None and src_ips is not None:
                output = model(data, masks, offsets, users, src_ips, dst_ips)
            elif users is not None and src_ips is None:
                output = model(data, masks, offsets, users)
            elif users is None and src_ips is not None:
                output = model(data, masks, offsets, src_ips, dst_ips)
            else:
                output = model(data, masks, offsets)
            loss = criterion(output, labels)

        loss.backward()

        #if not batch % 250:
        #    print(f"{output}\n{labels}")


        #total_norm = 0.0
        #for p in model.parameters():
        #    param_norm = p.grad.detach().data.norm(2)
        #    total_norm += param_norm.item() ** 2
        #total_norm = total_norm ** 0.5
        #print(total_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

        #total_norm = 0.0
        #for p in model.parameters():
        #    param_norm = p.grad.detach().data.norm(2)
        #    total_norm += param_norm.item() ** 2
        #total_norm = total_norm ** 0.5
        #print("\t" + str(total_norm))

        optimizer.step() 
        
        with torch.no_grad():
            if hasattr(model, "bins"):
                for b in model.bins:
                    b.clip()

        total_loss += loss.item()     

        if math.isnan(loss.item()):
            print("OOB grad")
            exit(0)

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | batch {batch:5d} / {total_batch_count:5d} | '
                  f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model, eval_data, batch_size, criterion, device="cpu", weights=None, return_all=False, ROC=False, strict=True, set_thresholds=-1, FSNet=False, CNNGRU=False, TypeNet=False):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    correct = 0.0
    total_count = 0
    total_user_list = []
    st_list = []
    real_user_list = []
    y_pred = []
    scores = []
    y_true = []
    cm_per_user = None 
    batch = 0
    auth = False
    total_batch_count = eval_data.batch_count(batch_size)

    if ROC or set_thresholds>=0:
        results_per_user = [[] for _ in range(model.n_users)]
        real_per_user = [[] for _ in range(model.n_users)]

    with torch.no_grad():
        while True:
            if not TypeNet:
                cont, data, target, masks, users, meta, offsets, src_ips, dst_ips = eval_data.get_batch(batch_size=batch_size, device=device, strict=strict)
            else:
                cont, data, target, positions, users, meta = eval_data.typenetbatch(batch_size=batch_size, device=device, strict=strict)

            if not cont:
                break
            batch += 1

            if batch % 50 == 0 and False:
                print(f"Eval batch {batch:5d} / {total_batch_count:5d}")

            if FSNet:
                # Run FSnet optim
                output, rec = model(data, users)
                local_y_pred = model.prediction(output, users)  
                #loss = (2*criterion(output, target)/np.log(2) + rec)/2
                loss = criterion(output, target)
            elif CNNGRU:
                output = model(data, users)
                loss = criterion(output, target) 
                local_y_pred = model.prediction(output, users)  
            elif TypeNet:
                output = model(data, users, positions)
                loss = criterion(output, target)
                local_y_pred = model.prediction(output, users)
            else:
                if users is not None and src_ips is not None:
                    output = model(data, masks, offsets, users, src_ips, dst_ips)
                    local_y_pred = model.prediction(output, users)
                elif users is not None and src_ips is None:
                    output = model(data, masks, offsets, users)
                    local_y_pred = model.prediction(output, users)
                elif users is None and src_ips is not None:
                    output = model(data, masks, offsets, src_ips, dst_ips)
                else:
                    output = model(data, masks, offsets)
                loss = criterion(output, target)


            correct +=  torch.sum(target.argmax(dim=1) == output.argmax(dim=1))
            total_count += output.shape[0]
            total_loss += output.shape[0] * loss.item()
            sm_output = F.softmax(output, dim=1)[:,0]
            try:
                scores.extend(sm_output.squeeze().data.cpu().numpy())
            except:
                scores.extend(sm_output.data.cpu().numpy())
            if users is None:
                y_pred.extend(torch.argmax(output, dim=1).data.cpu().numpy())
            else:
                y_pred.extend(local_y_pred.data.cpu().numpy())
            y_true.extend(torch.argmax(target.reshape(output.size(0), -1), dim=1).data.cpu().numpy())

            if users is not None:
                user_list = torch.argmax(users, dim=1)
                if ROC or set_thresholds>=0:
                    score = F.softmax(output, dim=1)[:,0]
                    real = torch.argmax(target.reshape(output.size(0), -1), dim=1)
                    for i in range(user_list.size(0)):
                        user = user_list[i].item()
                        results_per_user[user].append(score[i].item())
                        real_per_user[user].append(real[i].item())

                if return_all:
                    total_user_list.extend(eval_data.user_from_tensor(users))
                    real_user_list.extend([m['real_class'] for m in meta])
                    st_list.extend([m['st'] for m in meta])

                if cm_per_user is None:
                    cm_per_user = torch.tensor([[[0.0, 0.0],[0.0, 0.0]] for _ in range(model.n_users)]).to(device)

                auth = True
                real = (target.argmax(dim=1) == 1).float()
                pred = (output.argmax(dim=1) == 1).float()
                tn, fn = pred*real, pred*(1-real)
                fp, tp = (1-pred)*real, (1-pred)*(1-real)
                cm = torch.cat((tn.unsqueeze(1), fp.unsqueeze(1), fn.unsqueeze(1), tp.unsqueeze(1)), dim=1).reshape(-1, 2, 2)
                cm_per_user.index_add_(0, user_list, cm)


    if set_thresholds >= 0:
        for user_idx in range(len(results_per_user)):
            pos, neg = [], []
            for idx, score in enumerate(results_per_user[user_idx]):
                lbl = real_per_user[user_idx][idx]
                if lbl > 0:
                    pos.append(score)
                else:
                    neg.append(score)

            pos = sorted(pos)
            neg = sorted(neg)
            pos_idx = int(math.floor(set_thresholds*len(pos)))
            if pos_idx >= len(pos):
                pos_idx = len(pos) - 1
            if not len(pos):
                print(f"Why ? {eval_data.user_from_dim(user_idx)}. {len(neg)}")
            val = pos[pos_idx]
            thsd = (val + neg[binary_search(neg, val)]) / 2
            model.thresholds[user_idx] = thsd


    y_true_real = [str(eval_data.user_from_dim(int(t))) for t in y_true] 
    y_pred_real = [str(eval_data.user_from_dim(int(t))) for t in y_pred] 
    cf_matrix = confusion_matrix(y_true_real, y_pred_real, labels = [str(t) for t in eval_data.sorted_users])

    acc = correct / total_count
    if weights is not None:
        weight_vector = np.array([weights[y] for y in y_true])
        total_weight = np.ones_like(weight_vector).dot(weight_vector)
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        correct_weight = (y_pred == y_true).dot(weight_vector)
        acc_weighted = correct_weight / total_weight
    else:
        acc_weighted = acc

    # Get FP and TP per user
    if auth:
        counts = cm_per_user.sum(dim=1)
        filt = ((counts[:,0] * counts[:,1]) > 0)
        cm_per_user = cm_per_user[filt]
        counts = counts[filt]
        tfp = (cm_per_user[:,:,1] / counts).reshape(-1, 2)
    else:
        tfp = torch.tensor([[0.0, 0.0] for _ in range(len(cf_matrix))]).to(device)
        for u in range(len(cf_matrix)):
            if not cf_matrix[u,u]:
                continue
            tp = cf_matrix[u,u] / sum(cf_matrix[u,i] for i in range(len(cf_matrix)))
            fp = 1 - ( cf_matrix[u,u] / sum(cf_matrix[i,u] for i in range(len(cf_matrix))) )
            tfp[u,0] = fp
            tfp[u,1] = tp

    if ROC:
        full_roc = roc_curve(y_true, scores)
        per_user_curves = [None for _ in range(model.n_users)]
        for user in range(model.n_users):
            try:
                if sum(real_per_user[user]) == 0 or sum(real_per_user[user]) == len(real_per_user[user]):
                    continue
                per_user_curves[user] = roc_curve(real_per_user[user], results_per_user[user])
            except:
                pass
        if return_all:
            return total_loss / total_count, acc, acc_weighted, cf_matrix, tfp, total_user_list, y_true, y_pred, real_user_list, st_list, scores, full_roc, per_user_curves
        else:
            return total_loss / total_count, acc, acc_weighted, cf_matrix, tfp, full_roc, per_user_curves
    else:
        if return_all:
            return total_loss / total_count, acc, acc_weighted, cf_matrix, tfp, total_user_list, y_true, y_pred, real_user_list, st_list, scores
        else:
            return total_loss / total_count, acc, acc_weighted, cf_matrix, tfp

def predict(model, eval_data, batch_size, device="cpu", FSNet=False, CNNGRU=False, TypeNet=False):
    model.eval()  # turn on evaluation mode
    y_pred = []
    y_true = []
    y_time = []
    batch = 0
    total_batch_count = eval_data.batch_count(batch_size)

    with torch.no_grad():
        while True:
            if not TypeNet:
                cont, data, target, masks, users, meta, offsets, src_ips, dst_ips = eval_data.get_batch(batch_size=batch_size, device=device)
            else:
                cont, data, target, positions, users, meta = eval_data.typenetbatch(batch_size=batch_size, device=device)

            if not cont:
                break
            batch += 1

            if FSNet:
                # Run FSnet optim
                output, rec = model(data, users)
                local_y_pred = model.prediction(output, users)  
                #loss = (2*criterion(output, target)/np.log(2) + rec)/2
                loss = criterion(output, target)
            elif CNNGRU:
                output = model(data, users)
                loss = criterion(output, target) 
                local_y_pred = model.prediction(output, users)  
            elif TypeNet:
                output = model(data, users, positions)
                loss = criterion(output, target)
                local_y_pred = model.prediction(output, users)
            else:
                if users is not None and src_ips is not None:
                    output = model(data, masks, offsets, users, src_ips, dst_ips)
                    local_y_pred = model.prediction(output, users)
                elif users is not None and src_ips is None:
                    output = model(data, masks, offsets, users)
                    local_y_pred = model.prediction(output, users)
                elif users is None and src_ips is not None:
                    output = model(data, masks, offsets, src_ips, dst_ips)
                else:
                    output = model(data, masks, offsets)

            y_true.extend([m['real_class'] for m in meta])
            y_time.extend([m['st'] for m in meta])

            if users is None:
                new_data = torch.argmax(output, dim=1).data.cpu().numpy()
                y_pred.extend([eval_data.user_from_dim(t) for t in new_data])
            else:
                y_pred.extend([eval_data.user_from_dim(t) for t in local_y_pred.data.cpu().numpy()])


    return y_true, y_pred, y_time

def switch_device(device, *tensors):
    output = []
    for tensor in tensors:
        output.append(tensor.to(device))
    return tuple(output)


def save_conf_matrix(filename, m):
    with open(filename, "w") as outfile:
        outfile.write('\n'.join('\t'.join(str(a) for a in l) for l in m))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model_type, training_data, args):
    normalizer_params = []
    if args.S:
        normalizer_params.append(('lin', 1.0/1500))
    if args.T:
        normalizer_params.append(('tan', 1.0))

    if args.model_type == "FSNet":
        model = FSNet(training_data.user_count())
    elif args.model_type == "CNNGRU":
        model = CNNGRU(training_data.user_count())
    elif args.model_type == "TypeNet":
        model = TypeNet(training_data.user_count())
    elif args.model_type == "authentication":
        model = AuthenticationTransformer(training_data.user_count(), args.S + args.T, args.thead, args.tlayer, normalizer=normalizer_params, partition=(args.P > 0), partition_count=args.P, partition_size=args.p, d_pos=args.e, dropout=0.1, full=args.I)
    elif args.model_type == "authentication_alt":
        model = AuthenticationTransformer(training_data.user_count(), args.S + args.T, args.thead, args.tlayer, normalizer=normalizer_params, partition=(args.P > 0), partition_count=args.P, partition_size=args.p, d_pos=args.e, dropout=0.1, full=args.I, embed_lengths=True)
    else:
        model = ClassificationTransformer(training_data.user_count(), args.S + args.T, args.thead, args.tlayer, normalizer=normalizer_params, partition=(args.P > 0), partition_count=args.P, partition_size=args.p, d_pos=args.e, dropout=0.1, full=args.I)

    return model

def load_dataset(model_type, data_type, data_dir, args, eval_flag=False, reserve=False):
    model_condition = args.model_type == "classification" or args.model_type == "classification_without_others"
    type_condition  = args.data_type == "keystrokes_alternate"
    if not model_condition and type_condition:
        data = AlternateAuthenticationKeystrokeDataLoader(data_dir, \
                                                            max_run_length=args.N, \
                                                            min_run_length=args.n, \
                                                            dropout=0.0, \
                                                            use_lengths=False if not args.S else True,\
                                                            use_ips=args.I, \
                                                            evaluation=eval_flag, \
                                                            reserved_users=reserve)
    elif not model_condition and not type_condition:
        data = AuthenticationKeystrokeDataLoader(data_dir, \
                                                    run_length=args.N, \
                                                    use_ips=args.I, \
                                                    evaluation=eval_flag, \
                                                    reserved_users=reserve)
    elif model_condition and type_condition:
        data = AlternateAuthenticationKeystrokeDataLoader(data_dir, \
                                                            max_run_length=args.N, \
                                                            min_run_length=args.n, \
                                                            dropout=0.0 if eval_flag else 1.0, \
                                                            use_lengths=False if not args.S else True, \
                                                            use_ips=args.I, evaluation=eval_flag, \
                                                            include_others=args.model_type == "classification", \
                                                            reserved_users=reserve)
    else:
        data = ClassificationKeystrokeDataLoader(data_dir, \
                                                    run_length=args.N, \
                                                    use_ips=args.I, \
                                                    dropout=1.0 if eval_flag else 1.0, \
                                                    include_others=args.model_type == "classification", \
                                                    evaluation=eval_flag)

    return data


def main():
    parser = argparse.ArgumentParser(description='Run model')
    parser.add_argument('model_type', type=str, choices=["authentication", "classification", "FSNet", "CNNGRU", "classification_without_others", "TypeNet", "authentication_alt"], help="Type of model")
    parser.add_argument('data_type', type=str, help="Type of data : generic, keystrokes")
    parser.add_argument('data_dir', type=str, help="Dataset path")
    parser.add_argument('--testing-dir', type=str, nargs="*", default=[], help="Testing dataset path")
    parser.add_argument('--detailed-eval', default=0, action="count", help="Detailed evaluation")
    parser.add_argument('-N', type=int, default=512, help="Run length")
    parser.add_argument('-M', type=int, default=5120, help="Minimum number of runs per user", help=argparse.SUPPRESS)
    parser.add_argument('-B', type=int, default=32, help="Batch size")
    parser.add_argument('-P', type=int, default=8, help="Number of parititions")
    parser.add_argument('-p', type=int, default=8, help="Bins per partition")
    parser.add_argument('-e', type=int, default=-1, help="Positional encoding dimension")
    parser.add_argument('-L', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('-l', type=str, default=None, help="Model to load")
    parser.add_argument('-R', type=int, default=0, help="Randomness seed")
    parser.add_argument('-r', action="count", default=0, help="Reserve users")
    parser.add_argument('-c', type=int, default=5, help="Max tuple size - Eval")
    parser.add_argument('-S', type=int, default=1, help="Use packet sizes")
    parser.add_argument('-T', type=int, default=1, help="Use packet IAT")
    parser.add_argument('-E', type=int, default=15, help="Epoch count")
    parser.add_argument('--E1', type=int, default=-1, help="0 Dropout epochs")
    parser.add_argument('--E2', type=int, default=-1, help="Increasing dropout epochs")
    parser.add_argument('--thead', type=int, default=4, help="Transformer heads")
    parser.add_argument('--tlayer', type=int, default=4, help="Transformer layers")
    parser.add_argument('--threshold', type=float, default=0.98, help="TP threshold", help=argparse.SUPPRESS)
    parser.add_argument('--predict', action="count", default=0, help="Predict")
    parser.add_argument('-G', type=float, default=0.95, help="Gamma for LR decay")
    parser.add_argument('-H', type=int, default=5, help="Scheduler step")
    parser.add_argument('-O', type=str, default=None, help="Output file")
    parser.add_argument('-I', action="count", default=0, help="Use IPs")
    parser.add_argument('-g', type=int, default=-1, help="GPU number")
    parser.add_argument('-n', type=int, default=16, help="Min run length (for alternate loader)")
    args = parser.parse_args()

    if args.E1 == -1 and args.E2 == -1:
        args.E1 = args.E//3
        args.E2 = args.E1
    elif args.E1 == -1:
        args.E1 = min((args.E - args.E2)//2, 1)
    elif args.E2 == -1:
        args.E2 = min(args.E - 2*args.E1, 1)
    else:
        if args.E1 + args.E2 > args.E:
            print("Error : Subepochs need to be smaller than epoch count")
            exit(0)

    if args.O:
        output_dir = args.O
    else:
        name = ''.join(random.choices(string.ascii_lowercase, k=16))
        output_dir = "models/" + name
        try:
            os.mkdir(output_dir)
            print(f"Saving to directory {output_dir}")
        except:
            print(f"Directory {output_dir} exists !")
            exit()

    if args.R >= 0:
        torch.manual_seed(args.R)
        random.seed(args.R)
        np.random.seed(args.R)

    if args.g >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.g}')
    else:
        device = torch.device('cpu')

    eval_data_dir = args.data_dir + "/eval"
    train_data_dir = args.data_dir + "/train"
    training_data = load_dataset(args.model_type, args.data_type, train_data_dir, args)
    eval_data = load_dataset(args.model_type, args.data_type, eval_data_dir, args, eval_flag=True)
    model = load_model(args.model_type, training_data, args).to(device)
    n_users = training_data.user_count()
    best_val_loss, best_acc, best_acc_model, best_loss_model = float('inf'), 0.0, None, None

    print(f"You selected {args.model_type} : {args.data_type}. Parameters are : \n\tStride length : {args.N}"
            f"\n\tStride count per user : {args.M}"
            f"\n\tLearning rate : {args.L}"
            f"\n\tBatch size : {args.B}"
            f"\n\tUse IPs : {'True' if args.I else 'False'}"
            f"\n\tUse Lengths : {'True' if args.S else 'False'}"
            f"\n\tDevice : {device}\n\n"
            f"Dataset contains {n_users} users\n\n"
            f"Model (type {model.model_type}) contains {count_parameters(model)} parameters")

    if args.l:
        print("Loading model...")
        model = torch.load(args.l + "/model.pyt")
        model = model.to(device)
        best_acc_model = torch.load(args.l + "/model_acc.pyt")
        best_acc_model = model.to(device)
        best_loss_model = torch.load(args.l + "/model_loss.pyt")
        best_loss_model = model.to(device)
        print(f"Loaded model type {model.model_type}")


    if args.E >= 0:
        print("Training...")

        with open(output_dir + "/params", "w") as outfile:
            outfile.write(f"N\t{args.N}\nM\t{args.M}\nL\t{args.L}\nB\t{args.B}\nTraining\t{train_data_dir}\nValidation\t{eval_data_dir}\nTesting\t{args.testing_dir}\nUsers\t{n_users}\nParameters\t{count_parameters(model)}\nP\t{args.P}\np\t{args.p}\nE\t{args.E}\nmodel_type\t{args.model_type}\ndata_type\t{args.data_type}\nMin run length (optional)\t{args.n}\nUse IPs\t{'True' if args.I else 'False'}\nUse Lengths\t{'True' if args.S else 'False'}")

        lr = args.L
        epochs = args.E
        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.H, gamma=args.G)
        criterion = nn.CrossEntropyLoss(weight=training_data.weights(device=device))

        if epochs > 0:
            for epoch in range(1, epochs+1):
                epoch_start_time = time.time()
                criterion = nn.CrossEntropyLoss(weight=training_data.weights(device=device))
                #criterion = nn.CrossEntropyLoss()
                train(model, training_data, args.B, optimizer, scheduler, criterion, epoch=epoch-1, device=device, FSNet = args.model_type == "FSNet", TypeNet = args.model_type == "TypeNet", CNNGRU = args.model_type == "CNNGRU")
                training_data.reset()
                val_loss, acc, weighted_acc, conf_mat, _ = evaluate(model, eval_data, args.B, criterion, device=device, weights=training_data.weights().numpy(), strict=False, FSNet = args.model_type == "FSNet", TypeNet = args.model_type == "TypeNet", CNNGRU = args.model_type == "CNNGRU")
                eval_data.reset()
                val_ppl = math.exp(val_loss) 
                elapsed = time.time() - epoch_start_time 
                print('-' * 89)
                print(f'| End of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                      f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f} | acc {acc} | weighted_acc {weighted_acc}')
                print('-' * 89)

                if acc >= best_acc:
                    best_acc = acc
                    best_acc_model = copy.deepcopy(model)
                    torch.save(best_acc_model, output_dir+"/model_acc.pyt")

                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    best_loss_model = copy.deepcopy(model)
                    torch.save(best_loss_model, output_dir+"/model_loss.pyt")

                scheduler.step()
                if args.data_type == "keystrokes_alternate" and (args.model_type != "classification"):
                    training_data.set_dropout(max(0.0, min(1.0, (epoch-args.E1) / (args.E2))))
                    print(f"Set dropout to {training_data.dropout}")

        # Set tresholds
        #for pair in [(model, "model.pyt"), (best_acc_model, "model_acc.pyt"), (best_loss_model, "model_loss.pyt")]:
        #    m, name = pair
        #    if args.model_type != "classification":
        #        evaluate(m, training_data, args.B, criterion, device=device, set_thresholds=args.threshold, FSNet = args.model_type == "FSNet", TypeNet = args.model_type == "TypeNet", CNNGRU = args.model_type == "CNNGRU")
        #        training_data.reset()
        #    torch.save(m, output_dir+"/" + name)
                

    if args.predict:
        testing_data.set_dropout(-1)
        true, pred, times = predict(model, testing_data, args.B, device=device, FSNet = args.model_type == "FSNet", TypeNet = args.model_type == "TypeNet", CNNGRU = args.model_type == "CNNGRU")
        with open(output_dir+"/predictions.tsv", "w") as outfile:
            outfile.write("\n".join('{}\t{}\t{}'.format(times[i], true[i], pred[i]) for i in range(len(times))  ) )
        exit()


    # Eval
    datasets = []
    for m, m_type in ((model, "full"), (best_acc_model, "best_acc"), (best_loss_model, "best_loss")):
    #for m, m_type in ((model, "full"),):
        for idx, t in enumerate(args.testing_dir):
            datasets.append((f"{os.path.basename(t)}_{m_type}", t, m, True))
            #for u,_ in enumerate(m.thresholds):
            #    m.thresholds[u] = 0.5
        #datasets.append((f"validation_{m_type}", eval_data, m, False))


    if args.model_type == "FSNet":
        model.lenc.flatten_parameters()
        model.tenc.flatten_parameters()
        model.ldec.flatten_parameters()
        model.tdec.flatten_parameters()

    for name, data_raw, mod, load in datasets:
        print('=' * 89)
        print(name)
        print('=' * 89)
        
        if not load:
            data = data_raw
        else:
            data = load_dataset(args.model_type, args.data_type, data_raw, args, eval_flag=True, reserve=args.r)

        od = output_dir+"/"+name.replace(" ","_").lower()+"_results/"
        os.mkdir(od)
        tfps = [] 
        accuracies = []
        for u,_ in enumerate(mod.thresholds):
            mod.thresholds[u] = 0.5

        with open(od + "accuracies", "w") as outfile:
            # Evaluate on longer and longer sequences
            if args.data_type == "keystrokes":
                vals = range(1,args.c)
            elif args.data_type == "keystrokes_alternate":
                #vals = [-1] + list(reversed([int(round(i)) for i in np.linspace(args.n,args.N, 32)]))
                #vals = [args.N]
                vals = [-1, args.N, args.n]

            for eval_param in vals:
                if eval_param == -1 and args.data_type != "keystrokes_alternate":
                    print("Error")
                    exit()

                if args.data_type == "keystrokes_alternate" and eval_param != -1:
                    data.set_dropout(1 - ((eval_param - args.n) / (args.N - args.n)))
                elif eval_param == -1:
                    data.set_dropout(-1)

                criterion = nn.CrossEntropyLoss(weight=training_data.weights(device=device))
                #criterion = nn.CrossEntropyLoss()
                if args.detailed_eval and (args.model_type != "classification"):
                    val_loss, acc, weighted_acc, conf_mat, tfp, user_list, y_real, y_pred, real_user_list, st_list, scores, full_roc, per_user_roc = evaluate(mod, data, args.B, criterion, device=device, return_all=True, ROC=True, FSNet = args.model_type == "FSNet", TypeNet = args.model_type == "TypeNet", CNNGRU = args.model_type == "CNNGRU")
                elif args.detailed_eval:
                    val_loss, acc, weighted_acc, conf_mat, tfp, user_list, y_real, y_pred, real_user_list, st_list, scores = evaluate(mod, data, args.B, criterion, device=device, return_all=True, weights=training_data.weights().numpy())
                elif (args.model_type != "classification"):
                    val_loss, acc, weighted_acc, conf_mat, tfp, full_roc, per_user_roc = evaluate(mod, data, args.B, criterion, device=device, ROC=True, FSNet = args.model_type == "FSNet", TypeNet = args.model_type == "TypeNet", CNNGRU = args.model_type == "CNNGRU")
                else:
                    val_loss, acc, weighted_acc, conf_mat, tfp = evaluate(mod, data, args.B, criterion, device=device, weights=training_data.weights().numpy())

                val_ppl = math.exp(val_loss)
                print('-' * 89)
                if args.data_type == "keystrokes":
                    print(f'Using tuples of size {eval_param:3d} | '
                          f'loss {val_loss:5.2f} | ppl {val_ppl:8.2f} | acc {acc} | weighted_acc {weighted_acc}')
                else :
                    print(f'Using samples of length {eval_param} | '
                          f'loss {val_loss:5.2f} | ppl {val_ppl:8.2f} | acc {acc} | weighted_acc {weighted_acc}')
                print('-' * 89)
                if args.detailed_eval:
                    with open(od + f"predictions_{eval_param}.tsv", "w") as of:
                        of.write("\n".join("{}\t{}\t{}\t{}\t{}\t{}".format(st_list[i], user_list[i], real_user_list[i], y_real[i], y_pred[i], scores[i]) for i in range(len(st_list))))

                if args.model_type != "classification":
                    fpr, tpr, thsd = full_roc
                    plot_ROC(fpr, tpr, thsd, od + f"ROC_{eval_param}.png", f"tuples of size {eval_param}" if args.data_type == "keystrokes" else f"samples of length {eval_param}")
                    for user in range(len(per_user_roc)):
                        if per_user_roc[user] is not None:
                            fpr, tpr, thsd = per_user_roc[user]
                            user_lbl = data.user_from_dim(user)
                            plot_ROC(fpr, tpr, thsd, od + f"ROC_user{user_lbl}_{eval_param}.png", f"tuples of size {eval_param}" if args.data_type == "keystrokes" else f"samples of length {eval_param}", title=f"user #{user_lbl}")

                if args.data_type == "keystrokes":
                    data.set_tuple_size(tuple_size+1)

                data.reset()
                save_conf_matrix(od + f"conf_matrix_{eval_param}.tsv", conf_mat)
                outfile.write(f"{eval_param}\t{val_loss}\t{acc}\t{weighted_acc}\n")
                if eval_param != -1:
                    accuracies.append(acc)
                    if tfp is not None:
                        tfps.append(tfp)
            
        #plot_accuracy(accuracies, vals[1:], od + f"accuracy_per_tuplesize.png" if args.data_type == "keystrokes" else od + f"accuracy_per_dropout.png", "tuple size" if args.data_type == "keystrokes" else "sample size")
        #if len(tfps) and args.data_type == "keystrokes": 
        #    plot_tfp(tfps, od + "tfp_per_user.gif", lambda x: str(x+1) + " size tuples")
        #elif len(tfps):
        #    plot_tfp(tfps, od + "tfp_per_user.gif", lambda x: str(x+args.n) + " length samples")

main()
