import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas
import numpy as np
import glob
import os
from .logger import _logger


def log_confusion_matrices(cm_X_target, cm_X_pred, cm_id, comet_experiment, epoch):
    if comet_experiment:
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_target, title="Element to target", row_label="X", column_label="target", epoch=epoch, file_name="cm_X_target.json"
        )
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_pred, title="Element to pred", row_label="X", column_label="pred", epoch=epoch, file_name="cm_X_pred.json"
        )
        comet_experiment.log_confusion_matrix(
            matrix=cm_id, title="Target to pred", row_label="target", column_label="pred", epoch=epoch, file_name="cm_id.json"
        )


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

    if tensorboard_writer:
        sig_prob = torch.softmax(ypred_binary, axis=-1)[:, 1].to(torch.float32)
        for xcls in np.unique(X[:, 0]):
            fig = plt.figure()
            msk = X[:, 0] == xcls
            etarget = ytarget_flat[msk & (ytarget_flat[:, 0] != 0), 6]
            epred = ypred_p4[msk & (ypred_binary_cls != 0), 4]
            b = np.linspace(-2, 2, 100)
            plt.hist(etarget, bins=b, histtype="step", label="target")
            plt.hist(epred, bins=b, histtype="step", label="pred")
            plt.xlabel("log [E/E_elem]")
            plt.yscale("log")
            plt.legend(loc="best")
            tensorboard_writer.add_figure("energy_elemtype{}".format(int(xcls)), fig, global_step=epoch)

            fig = plt.figure()
            msk = X[:, 0] == xcls
            pt_target = ytarget_flat[msk & (ytarget_flat[:, 0] != 0), 2]
            pt_pred = ypred_p4[msk & (ypred_binary_cls != 0), 0]
            b = np.linspace(-2, 2, 100)  # Re-using b from energy plot, this is fine.
            plt.hist(pt_target, bins=b, histtype="step", label="target")
            plt.hist(pt_pred, bins=b, histtype="step", label="pred")
            plt.xlabel("log [pt/pt_elem]")
            plt.yscale("log")
            plt.legend(loc="best")
            tensorboard_writer.add_figure("pt_elemtype{}".format(int(xcls)), fig, global_step=epoch)

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

            fig = plt.figure()
            msk = X[:, 0] == xcls
            b = np.linspace(0, 1, 100)
            plt.hist(sig_prob[msk & (ytarget_flat[:, 0] == 0)], bins=b, histtype="step", label="target_noparticle")
            plt.hist(sig_prob[msk & (ytarget_flat[:, 0] != 0)], bins=b, histtype="step", label="target_particle")
            plt.xlabel("particle proba")
            plt.legend(loc="best")
            tensorboard_writer.add_figure("sig_proba_elemtype{}".format(int(xcls)), fig, global_step=epoch)

        try:
            tensorboard_writer.add_histogram("pt_target", torch.clamp(batch.ytarget[batch.mask][:, 2], -10, 10), global_step=epoch)
            tensorboard_writer.add_histogram("pt_pred", torch.clamp(ypred_raw[2][batch.mask][:, 0], -10, 10), global_step=epoch)
            ratio = (ypred_raw[2][batch.mask][:, 0] / batch.ytarget[batch.mask][:, 2])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("pt_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

            tensorboard_writer.add_histogram("eta_target", torch.clamp(batch.ytarget[batch.mask][:, 3], -10, 10), global_step=epoch)
            tensorboard_writer.add_histogram("eta_pred", torch.clamp(ypred_raw[2][batch.mask][:, 1], -10, 10), global_step=epoch)
            ratio = (ypred_raw[2][batch.mask][:, 1] / batch.ytarget[batch.mask][:, 3])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("eta_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

            tensorboard_writer.add_histogram("sphi_target", torch.clamp(batch.ytarget[batch.mask][:, 4], -10, 10), global_step=epoch)
            tensorboard_writer.add_histogram("sphi_pred", torch.clamp(ypred_raw[2][batch.mask][:, 2], -10, 10), global_step=epoch)
            ratio = (ypred_raw[2][batch.mask][:, 2] / batch.ytarget[batch.mask][:, 4])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("sphi_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

            tensorboard_writer.add_histogram("cphi_target", torch.clamp(batch.ytarget[batch.mask][:, 5], -10, 10), global_step=epoch)
            tensorboard_writer.add_histogram("cphi_pred", torch.clamp(ypred_raw[2][batch.mask][:, 3], -10, 10), global_step=epoch)
            ratio = (ypred_raw[2][batch.mask][:, 3] / batch.ytarget[batch.mask][:, 5])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("cphi_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

            tensorboard_writer.add_histogram("energy_target", torch.clamp(batch.ytarget[batch.mask][:, 6], -10, 10), global_step=epoch)
            tensorboard_writer.add_histogram("energy_pred", torch.clamp(ypred_raw[2][batch.mask][:, 4], -10, 10), global_step=epoch)
            ratio = (ypred_raw[2][batch.mask][:, 4] / batch.ytarget[batch.mask][:, 6])[batch.ytarget[batch.mask][:, 0] != 0]
            tensorboard_writer.add_histogram("energy_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)
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
        except ValueError as e:
            print(e)


def log_loss_correlation_plots(epoch_plot_data, epoch, writer, plot_prefix):
    """Logs 2D scatter plots of loss components vs total loss to TensorBoard."""
    for component_name, data_points in epoch_plot_data.items():
        if not data_points:
            _logger.warning(f"No data points for loss correlation plot: {plot_prefix}/{component_name} at epoch {epoch}")
            continue

        total_losses = [dp[0] for dp in data_points]
        component_values = [dp[1] for dp in data_points]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(total_losses, component_values, alpha=0.5, s=10)  # s=10 for smaller points
        ax.set_xlabel("Total Loss (batch-wise)")
        ax.set_ylabel(f"{component_name} Loss (batch-wise)")
        ax.set_title(f"{plot_prefix.capitalize()} Epoch {epoch}: {component_name} vs Total Loss")
        ax.grid(True)

        # Consider if log scale is needed for some losses, but scatter might be tricky with log
        # ax.set_xscale('log')
        # ax.set_yscale('log')

        try:
            writer.add_figure(f"epoch_loss_correlation/{plot_prefix}/{component_name}_vs_Total", fig, global_step=epoch)
        except Exception as e:
            _logger.error(f"Failed to log loss correlation plot {component_name} for {plot_prefix}: {e}")
        finally:
            plt.close(fig)


def log_epoch_loss_evolution_ratios(epoch_loss_history, first_epoch_in_history, current_epoch_logged, writer, plot_prefix, components_to_plot):
    """
    Logs scatter plots of (component_loss vs. total_loss) with each epoch as a marker to TensorBoard.
    Each plot shows the history up to current_epoch_logged.

    Args:
        epoch_loss_history (dict): Dict where keys are loss names and values are lists of losses per epoch.
        first_epoch_in_history (int): The actual epoch number corresponding to the first entry in history lists.
        current_epoch_logged (int): The current epoch number, used as global_step for TensorBoard.
                                   Also used to annotate the last point in the scatter plot.
        writer: TensorBoard SummaryWriter.
        plot_prefix (str): "train" or "valid".
        components_to_plot (list): List of component loss names to plot ratios for.
    """
    if not epoch_loss_history or "Total" not in epoch_loss_history or not epoch_loss_history["Total"]:
        _logger.warning(f"Not enough data in epoch_loss_history for evolution plots for {plot_prefix} at epoch {current_epoch_logged}.")
        return

    num_data_points = len(epoch_loss_history["Total"])
    if num_data_points == 0:
        return

    epochs_x_axis = list(range(first_epoch_in_history, first_epoch_in_history + num_data_points))
    total_losses_over_epochs = np.array(epoch_loss_history["Total"][:num_data_points])

    for component_name in components_to_plot:
        if component_name not in epoch_loss_history or len(epoch_loss_history.get(component_name, [])) < num_data_points:
            _logger.warning(
                f"Data for component {component_name} (len {len(epoch_loss_history.get(component_name, []))}) "
                f"is incomplete or missing for {plot_prefix} up to history length {num_data_points} (current epoch {current_epoch_logged})."
            )
            continue

        component_losses_over_epochs = np.array(epoch_loss_history[component_name][:num_data_points])

        fig, ax = plt.subplots(figsize=(10, 6))
        # Scatter plot of component loss vs total loss
        scatter = ax.scatter(total_losses_over_epochs, component_losses_over_epochs, marker="o", c=epochs_x_axis, cmap="viridis", alpha=0.7)
        ax.set_xlabel("Total Loss (epoch-wise)")
        ax.set_ylabel(f"{component_name} Loss (epoch-wise)")
        ax.set_title(f"{plot_prefix.capitalize()}: {component_name} vs Total Loss (Epoch Evolution)")
        ax.grid(True)

        # Add a colorbar to show epoch progression
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Epoch")

        # Annotate the last point with the current epoch number
        if num_data_points > 0:
            last_total_loss = total_losses_over_epochs[-1]
            last_component_loss = component_losses_over_epochs[-1]
            ax.annotate(
                f"Epoch {current_epoch_logged}",
                (last_total_loss, last_component_loss),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="red",
            )

        try:
            writer.add_figure(f"epoch_loss_evolution/{plot_prefix}/{component_name}_vs_Total_scatter", fig, global_step=current_epoch_logged)
        except Exception as e:
            _logger.error(f"Failed to log loss evolution scatter plot {component_name} for {plot_prefix}: {e}")
        finally:
            plt.close(fig)
