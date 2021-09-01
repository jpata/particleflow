import os
import pickle as pkl
import math, time, tqdm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

import torch

import pytorch_delphes

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

def compute_weights(gen_ids_one_hot, device, output_dim_id):
    vs, cs = torch.unique(gen_ids_one_hot, return_counts=True)
    weights = torch.zeros(output_dim_id).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0/math.sqrt(float(v))
    return weights

def make_plot_from_list(l, label, xlabel, ylabel, outpath, save_as):
    plt.style.use(hep.style.ROOT)

    if not os.path.exists(outpath + '/training_plots/'):
        os.makedirs(outpath + '/training_plots/')

    fig, ax = plt.subplots()
    ax.plot(range(len(l)), l, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    plt.savefig(outpath + '/training_plots/' + save_as + '.png')
    plt.close(fig)

    with open(outpath + '/training_plots/' + save_as + '.pkl', 'wb') as f:
        pkl.dump(l, f)

@torch.no_grad()
def test(model, multi_gpu, loader, epoch, alpha, target_type, device, output_dim_id, classification_only, outpath):
    with torch.no_grad():
        ret = train(model, multi_gpu, loader, epoch, None, alpha, target_type, device, output_dim_id, classification_only, outpath)
    return ret

def train(model, multi_gpu, loader, epoch, optimizer, alpha, target_type, device, output_dim_id, classification_only, outpath):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    #loss values for each batch: classification, regression, total
    losses_1, losses_2, losses_tot = [], [], []

    #accuracy values for each batch (monitor classification performance)
    accuracies_batch, accuracies_batch_msk = [], []

    #setup confusion matrix
    conf_matrix = np.zeros((output_dim_id, output_dim_id))

    # to compute average inference time
    t=[]

    for i, batch in enumerate(loader):
        t0 = time.time()

        if multi_gpu:
            X = batch
        else:
            X = batch.to(device)

        ## make like tensorflow model, 0-padding events to 6k elements
        # if X.x.shape[0]<6000:
        #     new_X = torch.cat([X.x,torch.zeros_like(X.x)[:6000-X.x.shape[0],:]])
        #     new_ygen_id = torch.cat([X.ygen_id,torch.zeros_like(X.ygen_id)[:6000-X.x.shape[0],:]])
        #     new_ygen_id[X.x.shape[0]:,0]=new_ygen_id[X.x.shape[0]:,0]+1
        #
        #     X.x = new_X
        #     X.ygen_id=new_ygen_id

        # Forwardprop
        if i<100:
            ti = time.time()
            pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4 = model(X)
            tf = time.time()
            if i!=0:
                t.append(round((tf-ti),2))
        else:
            pred_ids_one_hot, pred_p4, gen_ids_one_hot, gen_p4, cand_ids_one_hot, cand_p4 = model(X)

        _, gen_ids = torch.max(gen_ids_one_hot, -1)
        _, pred_ids = torch.max(pred_ids_one_hot, -1)
        _, cand_ids = torch.max(cand_ids_one_hot, -1)     # rule-based result

        # masking
        msk = ((pred_ids != 0) & (gen_ids != 0))
        msk2 = ((pred_ids != 0) & (pred_ids == gen_ids))

        # computing loss
        weights = compute_weights(torch.max(gen_ids_one_hot,-1)[1], device, output_dim_id)
        l1 = torch.nn.functional.cross_entropy(pred_ids_one_hot, gen_ids, weight=weights) # for classifying PID
        l2 = alpha * torch.nn.functional.mse_loss(pred_p4[msk2], gen_p4[msk2])  # for regressing p4

        if classification_only:
            loss = l1
        else:
            loss = l1+l2

        if is_train:
            # BACKPROP
            #print(list(model.parameters())[1].grad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses_1.append(l1.detach().cpu().item())
        losses_2.append(l2.detach().cpu().item())
        losses_tot.append(loss.detach().cpu().item())

        t1 = time.time()

        accuracies_batch.append(accuracy_score(gen_ids.detach().cpu().numpy(), pred_ids.detach().cpu().numpy()))
        accuracies_batch_msk.append(accuracy_score(gen_ids[msk].detach().cpu().numpy(), pred_ids[msk].detach().cpu().numpy()))

        conf_matrix += sklearn.metrics.confusion_matrix(gen_ids.detach().cpu().numpy(),
                                        np.argmax(pred_ids_one_hot.detach().cpu().numpy(),axis=1), labels=range(6))

        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(loader), loss.detach().cpu().item(), t1-t0), end='\r', flush=True)

    print("Average Inference time per event is: ", round((sum(t) / len(t)),2), 's')

    losses_1 = np.mean(losses_1)
    losses_2 = np.mean(losses_2)
    losses_tot = np.mean(losses_tot)

    acc = np.mean(accuracies_batch)
    acc_msk = np.mean(accuracies_batch_msk)

    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    return losses_tot, losses_1, losses_2, acc, acc_msk, conf_matrix, conf_matrix_norm


def train_loop(model, device, multi_gpu, train_loader, valid_loader, test_loader, n_epochs, patience, optimizer, alpha, target, output_dim_id, classification_only, outpath):
    t0_initial = time.time()

    losses_1_train, losses_2_train, losses_tot_train = [], [], []
    losses_1_valid, losses_2_valid, losses_tot_valid  = [], [], []

    accuracies_train, accuracies_msk_train = [], []
    accuracies_valid, accuracies_msk_valid = [], []

    best_val_loss = 99999.9
    stale_epochs = 0

    print("Training over {} epochs".format(n_epochs))
    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training epoch
        model.train()
        losses_tot, losses_1, losses_2, acc, acc_msk, conf_matrix, conf_matrix_norm = train(model, multi_gpu, train_loader, epoch, optimizer, alpha, target, device, output_dim_id, classification_only, outpath)

        losses_tot_train.append(losses_tot)
        losses_1_train.append(losses_1)
        losses_2_train.append(losses_2)

        accuracies_train.append(acc)
        accuracies_msk_train.append(acc_msk)

        # validation step
        model.eval()
        losses_tot_v, losses_1_v, losses_2_v, acc_v, acc_msk_v, conf_matrix_v, conf_matrix_norm_v = test(model, multi_gpu, valid_loader, epoch, alpha, target, device, output_dim_id, classification_only, outpath)

        losses_tot_valid.append(losses_tot_v)
        losses_1_valid.append(losses_1_v)
        losses_2_valid.append(losses_2_v)

        accuracies_valid.append(acc_v)
        accuracies_msk_valid.append(acc_msk_v)

        # early-stopping
        if losses_tot_v < best_val_loss:
            best_val_loss = losses_tot_v
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = n_epochs - (epoch+1)
        time_per_epoch = (t1 - t0_initial)/(epoch + 1)
        eta = epochs_remaining*time_per_epoch/60

        print("epoch={}/{} dt={:.2f}min train_loss={:.5f} valid_loss={:.5f} train_acc={:.5f} valid_acc={:.5f} train_acc_msk={:.5f} valid_acc_msk={:.5f} stale={} eta={:.1f}m".format(
            epoch+1, n_epochs,
            (t1-t0)/60, losses_tot_train[epoch], losses_tot_valid[epoch], accuracies_train[epoch], accuracies_valid[epoch],
            accuracies_msk_train[epoch], accuracies_msk_valid[epoch], stale_epochs, eta))

        torch.save(model.state_dict(), "{0}/epoch_{1}_weights.pth".format(outpath, epoch))

        torch.save(conf_matrix_norm, outpath + '/confusion_matrix_plots/cmT_normed_epoch_' + str(epoch) + '.pt')
        torch.save(conf_matrix_norm_v, outpath + '/confusion_matrix_plots/cmV_normed_epoch_' + str(epoch) + '.pkl')

    make_plot_from_list(losses_tot_train, 'train loss_tot', 'Epochs', 'Loss', outpath, 'losses_tot_train')
    make_plot_from_list(losses_1_train, 'train loss_1', 'Epochs', 'Loss', outpath, 'losses_1_train')
    make_plot_from_list(losses_2_train, 'train loss_2', 'Epochs', 'Loss', outpath, 'losses_2_train')

    make_plot_from_list(losses_tot_valid, 'valid loss_tot', 'Epochs', 'Loss', outpath, 'losses_tot_valid')
    make_plot_from_list(losses_1_valid, 'valid loss_1', 'Epochs', 'Loss', outpath, 'losses_1_valid')
    make_plot_from_list(losses_2_valid, 'valid loss_2', 'Epochs', 'Loss', outpath, 'losses_2_valid')

    make_plot_from_list(accuracies_train, 'train accuracy', 'Epochs', 'Accuracy', outpath, 'accuracies_train')
    make_plot_from_list(accuracies_msk_train, 'train accuracy_msk', 'Epochs', 'Accuracy', outpath, 'accuracies_msk_train')

    make_plot_from_list(accuracies_valid, 'valid accuracy', 'Epochs', 'Accuracy', outpath, 'accuracies_valid')
    make_plot_from_list(accuracies_msk_valid, 'valid accuracy_msk', 'Epochs', 'Accuracy', outpath, 'accuracies_msk_valid')

    print('Done with training.')
    return
