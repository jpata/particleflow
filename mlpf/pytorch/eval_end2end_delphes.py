#import setGPU
import torch
import torch_geometric
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
import pandas
import mplhep
import pickle

import graph_data_delphes
from graph_data_delphes import PFGraphDataset
from data_preprocessing import from_data_to_loader
import train_end2end_delphes
import time

elem_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_labels = [0, 1, 2, 3, 4, 5]

#map these to ids 0...Nclass
class_to_id = {r: class_labels[r] for r in range(len(class_labels))}

# map these to ids 0...Nclass
elem_to_id = {r: elem_labels[r] for r in range(len(elem_labels))}

def collate(items):
    l = sum(items, [])
    return Batch.from_data_list(l)

#Creates the dataframe of predictions given a trained model and a data loader
def prepare_dataframe(model, loader, multi_gpu, device):
    model.eval()
    dfs = []
    dfs_edges = []
    eval_time = 0

    for i, data in enumerate(loader):
        if not multi_gpu:
            data = data.to(device)
        pred_id_onehot, pred_momentum, new_edges = model(data)
        _, pred_id = torch.max(pred_id_onehot, -1)
        pred_momentum[pred_id==0] = 0
        data = [data]

        x = torch.cat([d.x.to("cpu") for d in data])
        gen_id = torch.cat([d.ygen_id.to("cpu") for d in data])
        gen_p4 = torch.cat([d.ygen[:, :].to("cpu") for d in data])
        cand_id = torch.cat([d.ycand_id.to("cpu") for d in data])
        cand_p4 = torch.cat([d.ycand[:, :].to("cpu") for d in data])

        # reverting the one_hot_embedding
        gen_id_flat = torch.max(gen_id, -1)[1]
        cand_id_flat = torch.max(cand_id, -1)[1]

        df = pandas.DataFrame()
        gen_p4.shape
        gen_id.shape

        # Recall:
        # [pid] takes from 1 to 6
        # [charge, pt (GeV), eta, sin phi, cos phi, E (GeV)]

        df["elem_type"] = [int(elem_labels[i]) for i in torch.argmax(x[:, :len(elem_labels)], axis=-1).numpy()]

        df["gen_pid"] = [int(class_labels[i]) for i in gen_id_flat.numpy()]
        df["gen_charge"] = gen_p4[:, 0].numpy()
        df["gen_eta"] = gen_p4[:, 2].numpy()
        df["gen_sin_phi"] = gen_p4[:, 3].numpy()
        df["gen_cos_phi"] = gen_p4[:, 4].numpy()
        df["gen_e"] = gen_p4[:, 5].numpy()

        df["cand_pid"] = [int(class_labels[i]) for i in cand_id_flat.numpy()]
        df["cand_charge"] = cand_p4[:, 0].numpy()
        df["cand_eta"] = cand_p4[:, 2].numpy()
        df["cand_sin_phi"] = cand_p4[:, 3].numpy()
        df["cand_cos_phi"] = cand_p4[:, 4].numpy()
        df["cand_e"] = cand_p4[:, 5].numpy()

        df["pred_pid"] = [int(class_labels[i]) for i in pred_id.detach().cpu().numpy()]
        df["pred_charge"] = pred_momentum[:, 0].detach().cpu().numpy()
        df["pred_eta"] = pred_momentum[:, 2].detach().cpu().numpy()
        df["pred_sin_phi"] = pred_momentum[:, 3].detach().cpu().numpy()
        df["pred_cos_phi"] = pred_momentum[:, 4].detach().cpu().numpy()
        df["pred_e"] = pred_momentum[:, 5].detach().cpu().numpy()

        dfs.append(df)
        #df_edges = pandas.DataFrame()
        #df_edges["edge0"] = edges[0].to("cpu")
        #df_edges["edge1"] = edges[1].to("cpu")
        #dfs_edges += [df_edges]

    df = pandas.concat(dfs, ignore_index=True)
    return df

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=sorted(train_end2end_delphes.model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--path", type=str, help="path to model", default="data/PFNet7_TTbar_14TeV_TuneCUETP8M1_cfi_gen__npar_221073__cfg_ee19d91068__user_jovyan__ntrain_400__lr_0.0001__1588215695")
    parser.add_argument("--epoch", type=str, default=0, help="Epoch to use")
    parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--start", type=int, default=3800, help="first file index to evaluate")
    parser.add_argument("--stop", type=int, default=4000, help="last file index to evaluate")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")

    epoch = args.epoch
    model = args.model
    path = args.path
    weights = torch.load("{}/epoch_{}_weights.pth".format(path, epoch), map_location=device)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}

    with open('{}/model_kwargs.pkl'.format(path),'rb') as f:
        model_kwargs = pickle.load(f)

    model_class = train_end2end_delphes.model_classes[args.model]
    model = model_class(**model_kwargs)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()


    print(args.dataset)
    full_dataset = PFGraphDataset(root=args.dataset)


    print("full_dataset", len(full_dataset))
    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=args.start, stop=args.stop))

    loader = DataListLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=True)
    loader.collate_fn = collate

    big_df = prepare_dataframe(model, loader, False, device)

    big_df.to_pickle("{}/df.pkl.bz2".format(path))
    #edges_df.to_csv("{}/edges.csv".format(path))
    print(big_df)
    #print(edges_df)
