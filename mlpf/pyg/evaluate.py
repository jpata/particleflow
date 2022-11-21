import glob
import time

import matplotlib
import numpy as np
import torch
import torch_geometric

from .utils import Y_FEATURES

matplotlib.use("Agg")


def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def make_predictions(rank, model, file_loader, batch_size, num_classes, PATH):
    """
    Runs inference on the qcd test dataset to evaluate performance.
    Saves the predictions as .pt files.
    Each .pt file will contain a dict() object with keys X, Y_pid, Y_p4;
    contains all the necessary event information to make plots.

    Args
        rank: int representing the gpu device id, or str=='cpu' (both work, trust me)
        model: pytorch model
        file_loader: a pytorch Dataloader that loads .pt files for training when you invoke the get() method
    """

    ti = time.time()

    t0, tf = time.time(), 0

    ibatch = 0
    for num, file in enumerate(file_loader):
        print(
            f"Time to load file {num+1}/{len(file_loader)} on rank {rank} is {round(time.time() - t0, 3)}s"
        )
        tf = tf + (time.time() - t0)

        file = [x for t in file for x in t]  # unpack the list of tuples to a list

        loader = torch_geometric.loader.DataLoader(file, batch_size=batch_size)

        t = 0
        for i, batch in enumerate(loader):

            t0 = time.time()
            pred_ids_one_hot, pred_p4 = model(batch.to(rank))
            t1 = time.time()
            # print(
            #     f"batch {i}/{len(loader)}, "
            #     f"forward pass on rank {rank} = {round(t1 - t0, 3)}s, "
            #     f"for batch with {batch.num_nodes} nodes"
            # )
            t = t + (t1 - t0)

            # zero pad the events to use the same plotting scripts as the tf pipeline
            padded_num_elem_size = 6400

            # must zero pad each event individually so must unpack the batches
            pred_ids_one_hot_list = []
            pred_p4_list = []
            for z in range(batch_size):
                pred_ids_one_hot_list.append(pred_ids_one_hot[batch.batch == z])
                pred_p4_list.append(pred_p4[batch.batch == z])

            X = []
            Y_pid = []
            Y_p4 = []
            batch_list = batch.to_data_list()
            for j, event in enumerate(batch_list):
                vars = {
                    "X": event.x.detach().to("cpu"),
                    "ygen": event.ygen.detach().to("cpu"),
                    "ycand": event.ycand.detach().to("cpu"),
                    "pred_p4": pred_p4_list[j].detach().to("cpu"),
                    "gen_ids_one_hot": one_hot_embedding(
                        event.ygen_id.detach().to("cpu"), num_classes
                    ),
                    "cand_ids_one_hot": one_hot_embedding(
                        event.ycand_id.detach().to("cpu"), num_classes
                    ),
                    "pred_ids_one_hot": pred_ids_one_hot_list[j].detach().to("cpu"),
                }

                vars_padded = {}
                for key, var in vars.items():
                    var = var[:padded_num_elem_size]
                    var = torch.nn.functional.pad(
                        var,
                        (0, 0, 0, padded_num_elem_size - var.shape[0]),
                        mode="constant",
                        value=0,
                    ).unsqueeze(0)
                    vars_padded[key] = var

                X.append(vars_padded["X"])
                Y_pid.append(
                    torch.cat(
                        [
                            vars_padded["gen_ids_one_hot"],
                            vars_padded["cand_ids_one_hot"],
                            vars_padded["pred_ids_one_hot"],
                        ]
                    ).unsqueeze(0)
                )
                Y_p4.append(
                    torch.cat(
                        [
                            vars_padded["ygen"],
                            vars_padded["ycand"],
                            vars_padded["pred_p4"],
                        ]
                    ).unsqueeze(0)
                )

            outfile = f"{PATH}/predictions/pred_batch{ibatch}_{rank}.pt"
            print(f"saving predictions at {outfile}")
            torch.save(
                {
                    "X": torch.cat(X),  # [batch_size, 6400, 41]
                    "Y_pid": torch.cat(Y_pid),  # [batch_size, 3, 6400, 41]
                    "Y_p4": torch.cat(Y_p4),
                },  # [batch_size, 3, 6400, 41]
                outfile,
            )

            ibatch += 1

        #     if i == 2:
        #         break
        # if num == 2:
        #     break

        print(
            f"Average inference time per batch on rank {rank} is {round((t / len(loader)), 3)}s"
        )

        t0 = time.time()

    print(
        f"Average time to load a file on rank {rank} is {round((tf / len(file_loader)), 3)}s"
    )

    print(
        f"Time taken to make predictions on rank {rank} is: {round(((time.time() - ti) / 60), 2)} min"
    )


def postprocess_predictions(pred_path):
    """
    Loads all the predictions .pt files and combines them after some necessary processing to make plots.
    Saves the processed predictions.
    """

    print("--> Concatenating all predictions...")
    t0 = time.time()

    Xs = []
    Y_pids = []
    Y_p4s = []

    PATH = list(glob.glob(f"{pred_path}/pred_batch*.pt"))
    for i, fi in enumerate(PATH):
        print(f"loading prediction # {i+1}/{len(PATH)}")
        dd = torch.load(fi)
        Xs.append(dd["X"])
        Y_pids.append(dd["Y_pid"])
        Y_p4s.append(dd["Y_p4"])

    Xs = torch.cat(Xs).numpy()
    Y_pids = torch.cat(Y_pids)
    Y_p4s = torch.cat(Y_p4s)

    # reformat the loaded files for convenient plotting
    yvals = {}
    yvals["gen_cls"] = Y_pids[:, 0, :, :].numpy()
    yvals["cand_cls"] = Y_pids[:, 1, :, :].numpy()
    yvals["pred_cls"] = Y_pids[:, 2, :, :].numpy()

    for feat, key in enumerate(Y_FEATURES):
        yvals[f"gen_{key}"] = Y_p4s[:, 0, :, feat].unsqueeze(-1).numpy()
        yvals[f"cand_{key}"] = Y_p4s[:, 1, :, feat].unsqueeze(-1).numpy()
        yvals[f"pred_{key}"] = Y_p4s[:, 2, :, feat].unsqueeze(-1).numpy()

    print(
        f"Time taken to concatenate all predictions is: {round(((time.time() - t0) / 60), 2)} min"
    )

    print("--> Further processing for convenient plotting")
    t0 = time.time()

    def flatten(arr):
        return arr.reshape(-1, arr.shape[-1])

    X_f = flatten(Xs)

    msk_X_f = X_f[:, 0] != 0

    for val in ["gen", "cand", "pred"]:
        yvals[f"{val}_phi"] = np.arctan2(
            yvals[f"{val}_sin_phi"], yvals[f"{val}_cos_phi"]
        )
        yvals[f"{val}_cls_id"] = np.argmax(yvals[f"{val}_cls"], axis=-1).reshape(
            yvals[f"{val}_cls"].shape[0], yvals[f"{val}_cls"].shape[1], 1
        )  # cz for some reason keepdims doesn't work

        yvals[f"{val}_px"] = np.sin(yvals[f"{val}_phi"]) * yvals[f"{val}_pt"]
        yvals[f"{val}_py"] = np.cos(yvals[f"{val}_phi"]) * yvals[f"{val}_pt"]

    yvals_f = {k: flatten(v) for k, v in yvals.items()}

    # remove the last dim
    for k in yvals_f.keys():
        if yvals_f[k].shape[-1] == 1:
            yvals_f[k] = yvals_f[k][..., -1]

    print(
        f"Time taken to process the predictions is: {round(((time.time() - t0) / 60), 2)} min"
    )

    print("-->Saving the processed events")
    t0 = time.time()
    torch.save(Xs, f"{pred_path}/post_processed_Xs.pt", pickle_protocol=4)
    torch.save(X_f, f"{pred_path}/post_processed_X_f.pt", pickle_protocol=4)
    torch.save(msk_X_f, f"{pred_path}/post_processed_msk_X_f.pt", pickle_protocol=4)
    torch.save(yvals, f"{pred_path}/post_processed_yvals.pt", pickle_protocol=4)
    torch.save(yvals_f, f"{pred_path}/post_processed_yvals_f.pt", pickle_protocol=4)
    print(
        f"Time taken to save the predictions is: {round(((time.time() - t0) / 60), 2)} min"
    )

    return Xs, X_f, msk_X_f, yvals, yvals_f
