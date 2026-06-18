import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas
import numpy as np
import glob
import os

from mlpf.logger import _logger


def log_confusion_matrices(cm_X_target, cm_X_pred, cm_id, comet_experiment, epoch):
    if comet_experiment:
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_target, title="Element to target", row_label="X", column_label="target", epoch=epoch, file_name="cm_X_target.json"
        )
        _logger.info("logged confusion matrix: Element to target")
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_pred, title="Element to pred", row_label="X", column_label="pred", epoch=epoch, file_name="cm_X_pred.json"
        )
        _logger.info("logged confusion matrix: Element to pred")
        comet_experiment.log_confusion_matrix(
            matrix=cm_id, title="Target to pred", row_label="target", column_label="pred", epoch=epoch, file_name="cm_id.json"
        )
        _logger.info("logged confusion matrix: Target to pred")


def log_oc_visualizations_to_tensorboard(batch, ypred_raw, ytarget, tensorboard_writer, step, dataset):
    from mlpf.model.utils import get_clustering

    # Select first event in batch
    event_idx = 0
    mask = batch.mask[event_idx]
    if mask.sum() == 0:
        return

    X = batch.X[event_idx][mask].cpu().numpy()
    ytarget_pn = ytarget["particle_number"][event_idx][mask].cpu().numpy()

    # Debug: Check if all PN are 0
    num_nonzero = np.sum(ytarget_pn > 0)
    _logger.info(f"OC visualization step {step}: event 0 has {mask.sum()} elements, {num_nonzero} non-zero PN in ground truth")

    if len(ypred_raw) <= 5:
        return

    oc_beta = ypred_raw[4][event_idx][mask].detach().cpu()
    oc_coords = ypred_raw[5][event_idx][mask].detach().cpu()

    pred_pn = get_clustering(oc_beta, oc_coords).cpu().numpy()

    # Clustering metrics comparing ground truth vs predicted cluster assignments
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    gt_labels = ytarget_pn.astype(np.int64)
    pred_labels = pred_pn.astype(np.int64)

    ari = adjusted_rand_score(gt_labels, pred_labels)
    nmi = normalized_mutual_info_score(gt_labels, pred_labels)

    # Hit-level signal/noise classification: GT (PN>0) vs pred (cluster!=-1)
    gt_signal = (gt_labels > 0).astype(np.int64)
    pred_signal = (pred_labels != -1).astype(np.int64)
    correct = (gt_signal == pred_signal).sum()
    total = len(gt_signal)
    signal_acc = correct / total

    _logger.info(
        f"OC metrics step {step}: ARI={ari:.4f} NMI={nmi:.4f} "
        f"signal_acc={signal_acc:.4f} "
        f"({num_nonzero} signal / {total} hits)"
    )
    tensorboard_writer.add_scalar("oc/ARI", ari, step)
    tensorboard_writer.add_scalar("oc/NMI", nmi, step)
    tensorboard_writer.add_scalar("oc/signal_accuracy", signal_acc, step)

    cmap = plt.get_cmap("tab20")

    elem_type = X[:, 0]
    pos = np.zeros((X.shape[0], 3))

    ds_name = dataset.value
    if "hits" in ds_name:
        pos = X[:, 6:9]
    elif ds_name in ["cld", "clic"]:
        # Clusters (type 2)
        msk_cl = elem_type == 2
        pos[msk_cl] = X[msk_cl, 6:9]
        # Tracks (type 1)
        msk_trk = elem_type == 1
        r_trk = X[msk_trk, 10]
        sphi_trk = X[msk_trk, 3]
        cphi_trk = X[msk_trk, 4]
        z0_trk = X[msk_trk, 14]
        pos[msk_trk] = np.stack([r_trk * cphi_trk, r_trk * sphi_trk, z0_trk], axis=1)
    elif ds_name == "cms":
        pos = X[:, 17:20]
    else:
        if X.shape[1] > 8:
            pos = X[:, 6:9]

    fig = plt.figure(figsize=(20, 10))

    for i, (pn, title) in enumerate([(ytarget_pn, "Ground Truth PN"), (pred_pn, "Predicted PN")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        # Map PN values to sequential indices so each unique value gets a distinct color
        unique_pn = np.unique(pn)
        pn_to_idx = {v: idx for idx, v in enumerate(unique_pn)}
        color_idx = np.array([pn_to_idx[v] for v in pn])

        for et in np.unique(elem_type):
            msk = elem_type == et
            marker = "s" if et == 2 else "o"
            size = 15 if et == 2 else 5
            alpha = 0.7 if et == 2 else 0.5
            label = "Type {}".format(int(et))
            ax.scatter(pos[msk, 0], pos[msk, 1], pos[msk, 2], c=color_idx[msk], cmap=cmap, vmin=0, vmax=max(len(unique_pn) - 1, 1), s=size, marker=marker, alpha=alpha, label=label)

        ax.set_title(title)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        if len(np.unique(elem_type)) > 1:
            ax.legend()

    tensorboard_writer.add_figure("oc_pn_visualization", fig, global_step=step)
    plt.close(fig)


def validation_plots(batch, ypred_raw, ytarget, ypred, tensorboard_writer, epoch, outdir):
    X = batch.X[batch.mask].cpu()
    ytarget_flat = batch.ytarget[batch.mask].cpu()
    ypred_binary = ypred_raw[0][batch.mask].detach().cpu()
    ypred_binary_cls = torch.argmax(ypred_binary, axis=-1)
    ypred_cls = ypred_raw[1][batch.mask].detach().cpu()
    ypred_p4 = ypred_raw[2][batch.mask].detach().cpu()

    to_concat = [X, ytarget_flat, ypred_binary, ypred_cls, ypred_p4]
    if len(ypred_raw) > 4:
        ypred_oc_beta = ypred_raw[4][batch.mask].detach().cpu()
        ypred_oc_coords = ypred_raw[5][batch.mask].detach().cpu()
        to_concat.append(ypred_oc_beta)
        to_concat.append(ypred_oc_coords)

    arr = torch.concatenate(
        to_concat,
        axis=-1,
    ).numpy()
    df = pandas.DataFrame(arr)
    df.to_parquet(f"{outdir}/batch0_epoch{epoch}.parquet")
    _logger.info(f"saved batch0_epoch{epoch}.parquet")

    if tensorboard_writer:
        if len(ypred_raw) > 4:
            fig = plt.figure()
            b = np.linspace(0, 1, 100)
            plt.hist(ypred_oc_beta.numpy(), bins=b, histtype="step")
            plt.xlabel("oc_beta")
            plt.yscale("log")
            tensorboard_writer.add_figure("oc_beta", fig, global_step=epoch)
            _logger.info("plotted oc_beta")

        sig_prob = torch.softmax(ypred_binary, axis=-1)[:, 1].to(torch.float32)
        for xcls in np.unique(X[:, 0]):
            fig = plt.figure()
            msk = X[:, 0] == xcls
            etarget = ytarget_flat[msk & (ytarget_flat[:, 0] != 0), 6]
            epred = ypred_p4[msk & (ypred_binary_cls != 0), 4]
            b = np.linspace(-2, 2, 100)
            plt.hist(etarget, bins=b, histtype="step")
            plt.hist(epred, bins=b, histtype="step")
            plt.xlabel("log [E/E_elem]")
            plt.yscale("log")
            tensorboard_writer.add_figure("energy_elemtype{}".format(int(xcls)), fig, global_step=epoch)
            _logger.info(f"plotted energy_elemtype{int(xcls)}")

            fig = plt.figure()
            msk = X[:, 0] == xcls
            pt_target = ytarget_flat[msk & (ytarget_flat[:, 0] != 0), 2]
            pt_pred = ypred_p4[msk & (ypred_binary_cls != 0), 0]
            b = np.linspace(-2, 2, 100)
            plt.hist(etarget, bins=b, histtype="step")
            plt.hist(epred, bins=b, histtype="step")
            plt.xlabel("log [pt/pt_elem]")
            plt.yscale("log")
            tensorboard_writer.add_figure("pt_elemtype{}".format(int(xcls)), fig, global_step=epoch)
            _logger.info(f"plotted pt_elemtype{int(xcls)}")

            fig = plt.figure(figsize=(5, 5))
            msk = (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            etarget = ytarget_flat[msk, 6]
            epred = ypred_p4[msk, 4]
            b = np.linspace(-2, 2, 100)
            plt.hist2d(etarget, epred, bins=b, cmap="hot", norm=matplotlib.colors.LogNorm())
            plt.plot([-4, 4], [-4, 4], color="black", ls="--")
            plt.xlabel("log [E_target/E_elem]")
            plt.ylabel("log [E_pred/E_elem]")
            tensorboard_writer.add_figure("energy_elemtype{}_corr".format(int(xcls)), fig, global_step=epoch)
            _logger.info(f"plotted energy_elemtype{int(xcls)}_corr")

            fig = plt.figure(figsize=(5, 5))
            msk = (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            pt_target = ytarget_flat[msk, 2]
            pt_pred = ypred_p4[msk, 0]
            b = np.linspace(-2, 2, 100)
            plt.hist2d(pt_target, pt_pred, bins=b, cmap="hot", norm=matplotlib.colors.LogNorm())
            plt.plot([-4, 4], [-4, 4], color="black", ls="--")
            plt.xlabel("log [pt_target/pt_elem]")
            plt.ylabel("log [pt_pred/pt_elem]")
            tensorboard_writer.add_figure("pt_elemtype{}_corr".format(int(xcls)), fig, global_step=epoch)
            _logger.info(f"plotted pt_elemtype{int(xcls)}_corr")

            fig = plt.figure(figsize=(5, 5))
            msk = (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            eta_target = ytarget_flat[msk, 3]
            eta_pred = ypred_p4[msk, 1]
            b = np.linspace(-6, 6, 100)
            plt.hist2d(eta_target, eta_pred, bins=b, cmap="hot", norm=matplotlib.colors.LogNorm())
            plt.plot([-6, 6], [-6, 6], color="black", ls="--")
            plt.xlabel("eta_target")
            plt.ylabel("eta_pred")
            tensorboard_writer.add_figure("eta_elemtype{}_corr".format(int(xcls)), fig, global_step=epoch)
            _logger.info(f"plotted eta_elemtype{int(xcls)}_corr")

            fig = plt.figure(figsize=(5, 5))
            msk = (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            sphi_target = ytarget_flat[msk, 4]
            sphi_pred = ypred_p4[msk, 2]
            b = np.linspace(-2, 2, 100)
            plt.hist2d(sphi_target, sphi_pred, bins=b, cmap="hot", norm=matplotlib.colors.LogNorm())
            plt.plot([-2, 2], [-2, 2], color="black", ls="--")
            plt.xlabel("sin_phi_target")
            plt.ylabel("sin_phi_pred")
            tensorboard_writer.add_figure("sphi_elemtype{}_corr".format(int(xcls)), fig, global_step=epoch)
            _logger.info(f"plotted sphi_elemtype{int(xcls)}_corr")

            fig = plt.figure(figsize=(5, 5))
            msk = (X[:, 0] == xcls) & (ytarget_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            cphi_target = ytarget_flat[msk, 5]
            cphi_pred = ypred_p4[msk, 3]
            b = np.linspace(-2, 2, 100)
            plt.hist2d(cphi_target, cphi_pred, bins=b, cmap="hot", norm=matplotlib.colors.LogNorm())
            plt.plot([-2, 2], [-2, 2], color="black", ls="--")
            plt.xlabel("cos_phi_target")
            plt.ylabel("cos_phi_pred")
            tensorboard_writer.add_figure("cphi_elemtype{}_corr".format(int(xcls)), fig, global_step=epoch)
            _logger.info(f"plotted cphi_elemtype{int(xcls)}_corr")

            fig = plt.figure()
            msk = X[:, 0] == xcls
            b = np.linspace(0, 1, 100)
            plt.hist(sig_prob[msk & (ytarget_flat[:, 0] == 0)], bins=b, histtype="step")
            plt.hist(sig_prob[msk & (ytarget_flat[:, 0] != 0)], bins=b, histtype="step")
            plt.xlabel("particle proba")
            tensorboard_writer.add_figure("sig_proba_elemtype{}".format(int(xcls)), fig, global_step=epoch)
            _logger.info(f"plotted sig_proba_elemtype{int(xcls)}")

        try:
            tensorboard_writer.add_histogram("pt_target", torch.clamp(batch.ytarget[batch.mask][:, 2], -10, 10), global_step=epoch)
            _logger.info("plotted pt_target histogram")
            tensorboard_writer.add_histogram("pt_pred", torch.clamp(ypred_raw[2][batch.mask][:, 0], -10, 10), global_step=epoch)
            _logger.info("plotted pt_pred histogram")
            ratio = (ypred_raw[2][batch.mask][:, 0] / batch.ytarget[batch.mask][:, 2])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("pt_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)
            _logger.info("plotted pt_ratio histogram")

            tensorboard_writer.add_histogram("eta_target", torch.clamp(batch.ytarget[batch.mask][:, 3], -10, 10), global_step=epoch)
            _logger.info("plotted eta_target histogram")
            tensorboard_writer.add_histogram("eta_pred", torch.clamp(ypred_raw[2][batch.mask][:, 1], -10, 10), global_step=epoch)
            _logger.info("plotted eta_pred histogram")
            ratio = (ypred_raw[2][batch.mask][:, 1] / batch.ytarget[batch.mask][:, 3])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("eta_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)
            _logger.info("plotted eta_ratio histogram")

            tensorboard_writer.add_histogram("sphi_target", torch.clamp(batch.ytarget[batch.mask][:, 4], -10, 10), global_step=epoch)
            _logger.info("plotted sphi_target histogram")
            tensorboard_writer.add_histogram("sphi_pred", torch.clamp(ypred_raw[2][batch.mask][:, 2], -10, 10), global_step=epoch)
            _logger.info("plotted sphi_pred histogram")
            ratio = (ypred_raw[2][batch.mask][:, 2] / batch.ytarget[batch.mask][:, 4])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("sphi_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)
            _logger.info("plotted sphi_ratio histogram")

            tensorboard_writer.add_histogram("cphi_target", torch.clamp(batch.ytarget[batch.mask][:, 5], -10, 10), global_step=epoch)
            _logger.info("plotted cphi_target histogram")
            tensorboard_writer.add_histogram("cphi_pred", torch.clamp(ypred_raw[2][batch.mask][:, 3], -10, 10), global_step=epoch)
            _logger.info("plotted cphi_pred histogram")
            ratio = (ypred_raw[2][batch.mask][:, 3] / batch.ytarget[batch.mask][:, 5])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("cphi_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)
            _logger.info("plotted cphi_ratio histogram")

            tensorboard_writer.add_histogram("energy_target", torch.clamp(batch.ytarget[batch.mask][:, 6], -10, 10), global_step=epoch)
            _logger.info("plotted energy_target histogram")
            tensorboard_writer.add_histogram("energy_pred", torch.clamp(ypred_raw[2][batch.mask][:, 4], -10, 10), global_step=epoch)
            _logger.info("plotted energy_pred histogram")
            ratio = (ypred_raw[2][batch.mask][:, 4] / batch.ytarget[batch.mask][:, 6])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("energy_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)
            _logger.info("plotted energy_ratio histogram")
        except ValueError as e:
            print(e)

        try:
            for attn in sorted(list(glob.glob(f"{outdir}/attn_conv_*.npz"))):
                attn_name = os.path.basename(attn).split(".")[0]
                attn_matrix = np.load(attn)["att"]
                batch_size = min(attn_matrix.shape[0], 8)
                fig, axes = plt.subplots(1, batch_size, figsize=((batch_size * 3, 1 * 3)))
                if isinstance(axes, matplotlib.axes._axes.Axes):
                    axes = [axes]
                for ibatch in range(batch_size):
                    plt.sca(axes[ibatch])
                    # plot the attention matrix of the first event in the batch
                    plt.imshow(attn_matrix[ibatch].T, cmap="hot", norm=matplotlib.colors.LogNorm())
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar()
                    plt.title("event {}, m={:.2E}".format(ibatch, np.mean(attn_matrix[ibatch][attn_matrix[ibatch] > 0])))
                plt.suptitle(attn_name)
                tensorboard_writer.add_figure(attn_name, fig, global_step=epoch)
                _logger.info(f"plotted attention matrix {attn_name}")
        except ValueError as e:
            print(e)
