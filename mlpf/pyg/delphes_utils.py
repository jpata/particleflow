import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from pyg.delphes_plots import (
    draw_efficiency_fakerate,
    plot_confusion_matrix,
    plot_distributions_all,
    plot_distributions_pid,
    plot_particle_multiplicity,
    plot_reso,
)
from pyg.utils import one_hot_embedding

matplotlib.use("Agg")


def make_predictions_delphes(model, multi_gpu, test_loader, outpath, device, epoch, num_classes):

    print("Making predictions...")
    t0 = time.time()

    gen_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "electron": [], "muon": []}
    pred_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "electron": [], "muon": []}
    cand_list = {"null": [], "chhadron": [], "nhadron": [], "photon": [], "electron": [], "muon": []}

    t = []

    for i, batch in enumerate(test_loader):
        if multi_gpu:
            X = batch  # a list (not torch) instance so can't be passed to device
        else:
            X = batch.to(device)

        ti = time.time()

        pred_ids_one_hot, pred_p4 = model(X)

        gen_p4 = X.ygen.detach().to("cpu")
        cand_ids_one_hot = one_hot_embedding(X.ycand_id.detach().to("cpu"), num_classes)
        gen_ids_one_hot = one_hot_embedding(X.ygen_id.detach().to("cpu"), num_classes)
        cand_p4 = X.ycand.detach().to("cpu")

        tf = time.time()
        if i != 0:
            t.append(round((tf - ti), 2))

        _, gen_ids = torch.max(gen_ids_one_hot.detach(), -1)
        _, pred_ids = torch.max(pred_ids_one_hot.detach(), -1)
        _, cand_ids = torch.max(cand_ids_one_hot.detach(), -1)

        # to make "num_gen vs num_pred" plots
        gen_list["null"].append((gen_ids == 0).sum().item())
        gen_list["chhadron"].append((gen_ids == 1).sum().item())
        gen_list["nhadron"].append((gen_ids == 2).sum().item())
        gen_list["photon"].append((gen_ids == 3).sum().item())
        gen_list["electron"].append((gen_ids == 4).sum().item())
        gen_list["muon"].append((gen_ids == 5).sum().item())

        pred_list["null"].append((pred_ids == 0).sum().item())
        pred_list["chhadron"].append((pred_ids == 1).sum().item())
        pred_list["nhadron"].append((pred_ids == 2).sum().item())
        pred_list["photon"].append((pred_ids == 3).sum().item())
        pred_list["electron"].append((pred_ids == 4).sum().item())
        pred_list["muon"].append((pred_ids == 5).sum().item())

        cand_list["null"].append((cand_ids == 0).sum().item())
        cand_list["chhadron"].append((cand_ids == 1).sum().item())
        cand_list["nhadron"].append((cand_ids == 2).sum().item())
        cand_list["photon"].append((cand_ids == 3).sum().item())
        cand_list["electron"].append((cand_ids == 4).sum().item())
        cand_list["muon"].append((cand_ids == 5).sum().item())

        gen_p4 = gen_p4.detach()
        pred_p4 = pred_p4.detach()
        cand_p4 = cand_p4.detach()

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

        if len(test_loader) < 5000:
            print(f"event #: {i+1}/{len(test_loader)}")
        else:
            print(f"event #: {i+1}/{5000}")

        if i == 4999:
            break

    print("Average Inference time per event is: ", round((sum(t) / len(t)), 2), "s")

    t1 = time.time()

    print("Time taken to make predictions is:", round(((t1 - t0) / 60), 2), "min")

    # store the 3 list dictionaries in a list (this is done only to compute the particle multiplicity plots)
    list = [pred_list, gen_list, cand_list]

    torch.save(list, outpath + "/list_for_multiplicities.pt")

    torch.save(gen_ids_all, outpath + "/gen_ids.pt")
    torch.save(gen_p4_all, outpath + "/gen_p4.pt")
    torch.save(pred_ids_all, outpath + "/pred_ids.pt")
    torch.save(pred_p4_all, outpath + "/pred_p4.pt")
    torch.save(cand_ids_all, outpath + "/cand_ids.pt")
    torch.save(cand_p4_all, outpath + "/cand_p4.pt")

    ygen = torch.cat([gen_ids_all.reshape(-1, 1).float(), gen_p4_all], axis=1)
    ypred = torch.cat([pred_ids_all.reshape(-1, 1).float(), pred_p4_all], axis=1)
    ycand = torch.cat([cand_ids_all.reshape(-1, 1).float(), cand_p4_all], axis=1)

    # store the actual predictions to make all the other plots
    predictions = {
        "ygen": ygen.reshape(1, -1, 7).detach().cpu().numpy(),
        "ycand": ycand.reshape(1, -1, 7).detach().cpu().numpy(),
        "ypred": ypred.detach().reshape(1, -1, 7).cpu().numpy(),
    }

    torch.save(predictions, outpath + "/predictions.pt")


def make_plots_delphes(model, test_loader, outpath, target, device, epoch, tag):

    print("Making plots...")
    t0 = time.time()

    # load the necessary predictions to make the plots
    gen_ids = torch.load(outpath + "/gen_ids.pt", map_location=device)
    gen_p4 = torch.load(outpath + "/gen_p4.pt", map_location=device)
    pred_ids = torch.load(outpath + "/pred_ids.pt", map_location=device)
    pred_p4 = torch.load(outpath + "/pred_p4.pt", map_location=device)
    cand_ids = torch.load(outpath + "/cand_ids.pt", map_location=device)
    cand_p4 = torch.load(outpath + "/cand_p4.pt", map_location=device)

    list_for_multiplicities = torch.load(outpath + "/list_for_multiplicities.pt", map_location=device)

    predictions = torch.load(outpath + "/predictions.pt", map_location=device)

    # reformat a bit
    ygen = predictions["ygen"].reshape(-1, 7)
    ypred = predictions["ypred"].reshape(-1, 7)
    ycand = predictions["ycand"].reshape(-1, 7)

    # make confusion matrix for MLPF
    target_names = ["none", "ch.had", "n.had", "g", "el", "mu"]
    conf_matrix_mlpf = sklearn.metrics.confusion_matrix(gen_ids.cpu(), pred_ids.cpu(), labels=range(6), normalize="true")

    plot_confusion_matrix(
        conf_matrix_mlpf, target_names, epoch, outpath + "/confusion_matrix_plots/", f"cm_mlpf_epoch_{str(epoch)}"
    )

    # make confusion matrix for rule based PF
    conf_matrix_cand = sklearn.metrics.confusion_matrix(gen_ids.cpu(), cand_ids.cpu(), labels=range(6), normalize="true")

    plot_confusion_matrix(
        conf_matrix_cand, target_names, epoch, outpath + "/confusion_matrix_plots/", "cm_cand", target="rule-based"
    )

    # making all the other plots
    if "QCD" in tag:
        sample = "QCD, 14 TeV, PU200"
    else:
        sample = "$t\\bar{t}$, 14 TeV, PU200"

    # make distribution plots
    plot_distributions_pid(
        1,
        gen_ids,
        gen_p4,
        pred_ids,
        pred_p4,
        cand_ids,
        cand_p4,  # distribution plots for chhadrons
        target,
        epoch,
        outpath,
        legend_title=sample + "\n",
    )
    plot_distributions_pid(
        2,
        gen_ids,
        gen_p4,
        pred_ids,
        pred_p4,
        cand_ids,
        cand_p4,  # distribution plots for nhadrons
        target,
        epoch,
        outpath,
        legend_title=sample + "\n",
    )
    plot_distributions_pid(
        3,
        gen_ids,
        gen_p4,
        pred_ids,
        pred_p4,
        cand_ids,
        cand_p4,  # distribution plots for photons
        target,
        epoch,
        outpath,
        legend_title=sample + "\n",
    )
    plot_distributions_pid(
        4,
        gen_ids,
        gen_p4,
        pred_ids,
        pred_p4,
        cand_ids,
        cand_p4,  # distribution plots for electrons
        target,
        epoch,
        outpath,
        legend_title=sample + "\n",
    )
    plot_distributions_pid(
        5,
        gen_ids,
        gen_p4,
        pred_ids,
        pred_p4,
        cand_ids,
        cand_p4,  # distribution plots for muons
        target,
        epoch,
        outpath,
        legend_title=sample + "\n",
    )

    plot_distributions_all(
        gen_ids,
        gen_p4,
        pred_ids,
        pred_p4,
        cand_ids,
        cand_p4,  # distribution plots for all together
        target,
        epoch,
        outpath,
        legend_title=sample + "\n",
    )

    # plot particle multiplicity plots
    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    plot_particle_multiplicity(list_for_multiplicities, "null", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_null.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_null.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    plot_particle_multiplicity(list_for_multiplicities, "chhadron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_chhadron.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_chhadron.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    plot_particle_multiplicity(list_for_multiplicities, "nhadron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_nhadron.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_nhadron.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    plot_particle_multiplicity(list_for_multiplicities, "photon", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_photon.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_photon.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    plot_particle_multiplicity(list_for_multiplicities, "electron", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_electron.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_electron.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2 * 8))
    plot_particle_multiplicity(list_for_multiplicities, "muon", ax)
    plt.savefig(outpath + "/multiplicity_plots/num_muon.png", bbox_inches="tight")
    plt.savefig(outpath + "/multiplicity_plots/num_muon.pdf", bbox_inches="tight")
    plt.close(fig)

    # make efficiency and fake rate plots for charged hadrons
    ax, _ = draw_efficiency_fakerate(
        ygen,
        ypred,
        ycand,
        1,
        "pt",
        np.linspace(0, 3, 61),
        outpath + "/efficiency_plots/eff_fake_pid1_pt.png",
        both=True,
        legend_title=sample + "\n",
    )
    ax, _ = draw_efficiency_fakerate(
        ygen,
        ypred,
        ycand,
        1,
        "eta",
        np.linspace(-3, 3, 61),
        outpath + "/efficiency_plots/eff_fake_pid1_eta.png",
        both=True,
        legend_title=sample + "\n",
    )
    ax, _ = draw_efficiency_fakerate(
        ygen,
        ypred,
        ycand,
        1,
        "energy",
        np.linspace(0, 50, 75),
        outpath + "/efficiency_plots/eff_fake_pid1_energy.png",
        both=True,
        legend_title=sample + "\n",
    )

    # make efficiency and fake rate plots for neutral hadrons
    ax, _ = draw_efficiency_fakerate(
        ygen,
        ypred,
        ycand,
        2,
        "pt",
        np.linspace(0, 3, 61),
        outpath + "/efficiency_plots/eff_fake_pid2_pt.png",
        both=True,
        legend_title=sample + "\n",
    )
    ax, _ = draw_efficiency_fakerate(
        ygen,
        ypred,
        ycand,
        2,
        "eta",
        np.linspace(-3, 3, 61),
        outpath + "/efficiency_plots/eff_fake_pid2_eta.png",
        both=True,
        legend_title=sample + "\n",
    )
    ax, _ = draw_efficiency_fakerate(
        ygen,
        ypred,
        ycand,
        2,
        "energy",
        np.linspace(0, 50, 75),
        outpath + "/efficiency_plots/eff_fake_pid2_energy.png",
        both=True,
        legend_title=sample + "\n",
    )

    # make resolution plots for chhadrons: pid=1
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 1, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 1, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 1, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid1_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid1_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for nhadrons: pid=2
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 2, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 2, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 2, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid2_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid2_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for photons: pid=3
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 3, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 3, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 3, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid3_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid3_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for electrons: pid=4
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 4, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 4, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 4, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid4_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid4_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    # make resolution plots for muons: pid=5
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 5, "pt", 2, ax=ax1, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_pt.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_pt.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 5, "eta", 0.2, ax=ax2, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_eta.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_eta.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    fig, (ax3) = plt.subplots(1, 1, figsize=(8, 8))
    plot_reso(ygen, ypred, ycand, 5, "energy", 0.2, ax=ax3, legend_title=sample + "\n")
    plt.savefig(outpath + "/resolution_plots/res_pid5_energy.png", bbox_inches="tight")
    plt.savefig(outpath + "/resolution_plots/res_pid5_energy.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.close(fig)

    t1 = time.time()
    print("Time taken to make plots is:", round(((t1 - t0) / 60), 2), "min")
