from pyg.utils_plots import plot_confusion_matrix
from pyg.utils_plots import plot_distributions_pid, plot_distributions_all, plot_particle_multiplicity
from pyg.utils_plots import draw_efficiency_fakerate, plot_reso
from pyg.utils_plots import pid_to_name_delphes, name_to_pid_delphes, pid_to_name_cms
from pyg.utils import define_regions, batch_event_into_regions

import torch
from torch_geometric.data import Batch
import mplhep as hep
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import math
import time
import tqdm
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42


def make_predictions(device, data, batch_events, output_dim_id, model, multi_gpu, test_loader, outpath):

    print('Making predictions...')

    gen_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "ele": [], "mu": []}
    pred_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "ele": [], "mu": []}
    cand_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "ele": [], "mu": []}

    if batch_events:    # batch events into eta,phi regions to build graphs only within regions
        regions = define_regions(num_eta_regions=10, num_phi_regions=10)

    t = 0
    t0 = time.time()
    for i, batch in enumerate(test_loader):

        if batch_events:    # batch events into eta,phi regions to build graphs only within regions
            batch = batch_event_into_regions(batch, regions)

        if multi_gpu:
            X = batch   # a list (not torch) instance so can't be passed to device
        else:
            X = batch.to(device)

        ti = time.time()
        pred, target, _, _ = model(X)
        tf = time.time()
        t = t + (tf - ti)

        # retrieve target
        gen_ids_one_hot = target['ygen_id']
        gen_p4 = target['ygen'].detach()
        cand_ids_one_hot = target['ycand_id']
        cand_p4 = target['ycand'].detach()

        # retrieve predictions
        pred_ids_one_hot = pred[:, :output_dim_id]
        pred_p4 = pred[:, output_dim_id:].detach()

        # revert on-hot encodings
        _, gen_ids = torch.max(gen_ids_one_hot.detach(), -1)
        _, pred_ids = torch.max(pred_ids_one_hot.detach(), -1)
        _, cand_ids = torch.max(cand_ids_one_hot.detach(), -1)

        # to make "num_gen vs num_pred" plots
        if data == 'delphes':
            dict = name_to_pid_delphes
        elif data == 'cms':
            dict = name_to_pid_cms

        for key, value in dict.items():
            gen_list[key].append((gen_ids == value).sum().item())
            pred_list[key].append((pred_ids == value).sum().item())
            cand_list[key].append((cand_ids == value).sum().item())

        if i == 0:
            gen_ids_all = gen_ids
            gen_p4_all = gen_p4

            pred_ids_all = pred_ids
            pred_p4_all = pred_p4

            cand_ids_all = cand_ids
            cand_p4_all = cand_p4
        else:
            gen_ids_all = torch.cat([gen_ids_all, gen_ids])
            gen_p4_all = torch.cat([gen_p4_all, gen_p4])

            pred_ids_all = torch.cat([pred_ids_all, pred_ids])
            pred_p4_all = torch.cat([pred_p4_all, pred_p4])

            cand_ids_all = torch.cat([cand_ids_all, cand_ids])
            cand_p4_all = torch.cat([cand_p4_all, cand_p4])

        print(f'event #: {i+1}/{len(test_loader)}')

        if i == 3:
            break

    print(f'Average inference time per event is {round((t / len(test_loader)),3)}s')

    print('Time taken to make predictions is:', round(((time.time() - t0) / 60), 2), 'min')

    # store the 3 list dictionaries in a list (this is done only to compute the particle multiplicity plots)
    if data == 'delphes':
        list = [pred_list, gen_list, cand_list]
        torch.save(list, outpath + '/list_for_multiplicities.pt')

    torch.save(gen_ids_all, outpath + '/gen_ids.pt')
    torch.save(gen_p4_all, outpath + '/gen_p4.pt')
    torch.save(pred_ids_all, outpath + '/pred_ids.pt')
    torch.save(pred_p4_all, outpath + '/pred_p4.pt')
    torch.save(cand_ids_all, outpath + '/cand_ids.pt')
    torch.save(cand_p4_all, outpath + '/cand_p4.pt')

    ygen = torch.cat([gen_ids_all.reshape(-1, 1).float(), gen_p4_all], axis=1)
    ypred = torch.cat([pred_ids_all.reshape(-1, 1).float(), pred_p4_all], axis=1)
    ycand = torch.cat([cand_ids_all.reshape(-1, 1).float(), cand_p4_all], axis=1)

    # store the actual predictions to make all the other plots
    predictions = {"ygen": ygen.reshape(1, -1, 7).detach().cpu().numpy(),
                   "ycand": ycand.reshape(1, -1, 7).detach().cpu().numpy(),
                   "ypred": ypred.detach().reshape(1, -1, 7).cpu().numpy()}

    torch.save(predictions, outpath + '/predictions.pt')


def make_plots(data, output_dim_id, model, test_loader, outpath, target, device, epoch, tag):

    print('Making plots...')
    t0 = time.time()

    # load the necessary predictions to make the plots
    gen_ids = torch.load(outpath + f'/gen_ids.pt', map_location=device)
    gen_p4 = torch.load(outpath + f'/gen_p4.pt', map_location=device)
    pred_ids = torch.load(outpath + f'/pred_ids.pt', map_location=device)
    pred_p4 = torch.load(outpath + f'/pred_p4.pt', map_location=device)
    cand_ids = torch.load(outpath + f'/cand_ids.pt', map_location=device)
    cand_p4 = torch.load(outpath + f'/cand_p4.pt', map_location=device)

    predictions = torch.load(outpath + f'/predictions.pt', map_location=device)

    # reformat a bit
    ygen = predictions["ygen"].reshape(-1, 7)
    ypred = predictions["ypred"].reshape(-1, 7)
    ycand = predictions["ycand"].reshape(-1, 7)

    # make confusion matrix for MLPF
    if data == 'delphes':
        target_names = ["none", "ch.had", "n.had", "g", "el", "mu"]
    elif data == 'cms':
        target_names = ["none", "HFEM", "HFHAD", "el", "mu", "g", "n.had", "ch.had"]

    conf_matrix_mlpf = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        pred_ids.cpu(),
                                                        labels=range(output_dim_id), normalize="true")

    plot_confusion_matrix(conf_matrix_mlpf, target_names, epoch + 1, outpath + '/confusion_matrix_plots/', f'cm_mlpf_epoch_{str(epoch)}')

    # make confusion matrix for rule based PF
    conf_matrix_cand = sklearn.metrics.confusion_matrix(gen_ids.cpu(),
                                                        cand_ids.cpu(),
                                                        labels=range(output_dim_id), normalize="true")

    plot_confusion_matrix(conf_matrix_cand, target_names, epoch + 1, outpath + '/confusion_matrix_plots/', 'cm_cand', target="rule-based")

    # making all the other plots
    if 'QCD' in tag:
        sample = "QCD, 14 TeV, PU200"
    else:
        sample = "$t\\bar{t}$, 14 TeV, PU200"

    # make distribution plots
    plot_distributions_pid(1, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for chhadrons
                           target, epoch, outpath, legend_title=sample + "\n")
    plot_distributions_pid(2, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for nhadrons
                           target, epoch, outpath, legend_title=sample + "\n")
    plot_distributions_pid(3, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for photons
                           target, epoch, outpath, legend_title=sample + "\n")
    plot_distributions_pid(4, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for electrons
                           target, epoch, outpath, legend_title=sample + "\n")
    plot_distributions_pid(5, gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for muons
                           target, epoch, outpath, legend_title=sample + "\n")

    plot_distributions_all(gen_ids, gen_p4, pred_ids, pred_p4, cand_ids, cand_p4,    # distribution plots for all together
                           target, epoch, outpath, legend_title=sample + "\n")

    # plot particle multiplicity plots
    if data == 'delphes':
        list_for_multiplicities = torch.load(outpath + f'/list_for_multiplicities.pt', map_location=device)

        fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
        ret_num_particles_null = plot_particle_multiplicity(list_for_multiplicities, "null", ax)
        plt.savefig(outpath + "/multiplicity_plots/num_null.png", bbox_inches="tight")
        plt.savefig(outpath + "/multiplicity_plots/num_null.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
        ret_num_particles_chhad = plot_particle_multiplicity(list_for_multiplicities, "chhadron", ax)
        plt.savefig(outpath + "/multiplicity_plots/num_chhadron.png", bbox_inches="tight")
        plt.savefig(outpath + "/multiplicity_plots/num_chhadron.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
        ret_num_particles_nhad = plot_particle_multiplicity(list_for_multiplicities, "nhadron", ax)
        plt.savefig(outpath + "/multiplicity_plots/num_nhadron.png", bbox_inches="tight")
        plt.savefig(outpath + "/multiplicity_plots/num_nhadron.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
        ret_num_particles_photon = plot_particle_multiplicity(list_for_multiplicities, "photon", ax)
        plt.savefig(outpath + "/multiplicity_plots/num_photon.png", bbox_inches="tight")
        plt.savefig(outpath + "/multiplicity_plots/num_photon.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
        ret_num_particles_electron = plot_particle_multiplicity(list_for_multiplicities, "ele", ax)
        plt.savefig(outpath + "/multiplicity_plots/num_electron.png", bbox_inches="tight")
        plt.savefig(outpath + "/multiplicity_plots/num_electron.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
        ret_num_particles_muon = plot_particle_multiplicity(list_for_multiplicities, "mu", ax)
        plt.savefig(outpath + "/multiplicity_plots/num_muon.png", bbox_inches="tight")
        plt.savefig(outpath + "/multiplicity_plots/num_muon.pdf", bbox_inches="tight")
        plt.close(fig)

    # make efficiency and fake rate plots for charged hadrons
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "pt", np.linspace(0, 3, 61), outpath + "/efficiency_plots/eff_fake_pid1_pt.png", both=True, legend_title=sample + "\n")
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "eta", np.linspace(-3, 3, 61), outpath + "/efficiency_plots/eff_fake_pid1_eta.png", both=True, legend_title=sample + "\n")
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 1, "energy", np.linspace(0, 50, 75), outpath + "/efficiency_plots/eff_fake_pid1_energy.png", both=True, legend_title=sample + "\n")

    # make efficiency and fake rate plots for neutral hadrons
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "pt", np.linspace(0, 3, 61), outpath + "/efficiency_plots/eff_fake_pid2_pt.png", both=True, legend_title=sample + "\n")
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "eta", np.linspace(-3, 3, 61), outpath + "/efficiency_plots/eff_fake_pid2_eta.png", both=True, legend_title=sample + "\n")
    ax, _ = draw_efficiency_fakerate(ygen, ypred, ycand, 2, "energy", np.linspace(0, 50, 75), outpath + "/efficiency_plots/eff_fake_pid2_energy.png", both=True, legend_title=sample + "\n")

    # make resolution plots for chhadrons: pid=1
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_pt = plot_reso(ygen, ypred, ycand, 1, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_eta = plot_reso(ygen, ypred, ycand, 1, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_chhad_E = plot_reso(ygen, ypred, ycand, 1, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for nhadrons: pid=2
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_pt = plot_reso(ygen, ypred, ycand, 2, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_eta = plot_reso(ygen, ypred, ycand, 2, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_nhad_E = plot_reso(ygen, ypred, ycand, 2, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for photons: pid=3
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_pt = plot_reso(ygen, ypred, ycand, 3, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_eta = plot_reso(ygen, ypred, ycand, 3, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_photon_E = plot_reso(ygen, ypred, ycand, 3, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for electrons: pid=4
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_pt = plot_reso(ygen, ypred, ycand, 4, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_eta = plot_reso(ygen, ypred, ycand, 4, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_electron_E = plot_reso(ygen, ypred, ycand, 4, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for muons: pid=5
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_pt = plot_reso(ygen, ypred, ycand, 5, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_eta = plot_reso(ygen, ypred, ycand, 5, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    res_muon_E = plot_reso(ygen, ypred, ycand, 5, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    t1 = time.time()
    print('Time taken to make plots is:', round(((t1 - t0) / 60), 2), 'min')
