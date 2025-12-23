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


def validation_plots(batch, ypred_raw, ytarget, ypred, tensorboard_writer, epoch, outdir):
    X = batch.X[batch.mask].cpu()
    ytarget_flat = batch.ytarget[batch.mask].cpu()
    ypred_binary = ypred_raw[0][batch.mask].detach().cpu()
    ypred_binary_cls = torch.argmax(ypred_binary, axis=-1)
    ypred_cls = ypred_raw[1][batch.mask].detach().cpu()
    ypred_p4 = ypred_raw[2][batch.mask].detach().cpu()

    arr = torch.concatenate(
        [X, ytarget_flat, ypred_binary, ypred_cls, ypred_p4],
        axis=-1,
    ).numpy()
    df = pandas.DataFrame(arr)
    df.to_parquet(f"{outdir}/batch0_epoch{epoch}.parquet")
    _logger.info(f"saved batch0_epoch{epoch}.parquet")

    if tensorboard_writer:
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
