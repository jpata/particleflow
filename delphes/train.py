import setGPU
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SGConv, GATConv
import numpy as np
import time
from torch_geometric.data import Data, DataLoader, Dataset
import os.path as osp

use_gpu = True
device = torch.device('cuda' if use_gpu else 'cpu')

map_candid_to_pdgid = {
    0: [0],
    211: [211, 2212, 321, -3112, 3222, -3312, -3334],
    -211: [-211, -2212, -321, 3112, -3222, 3312, 3334],
    130: [130, 2112, -2112, 310, 3122, -3122, 3322, -3322],
    22: [22],
    11: [11],
    -11: [-11],
     13: [13],
     -13: [-13]
}

map_candid_to_numid = {
    0: 0,
    211: 1,
    -211: 2,
    130: 3,
    22: 4,
    11: 5,
    -11: 6,
    13: 7,
    -13: 8,
}

def encode_ids(src_ids, map_candid_to_pdgid, map_candid_to_numid):
    new_ids = torch.zeros_like(src_ids)
    new_ids[:] = -1
    
    for candid, pdgids in map_candid_to_pdgid.items():
        numid = map_candid_to_numid[candid]
        for pdgid in pdgids:
            m = src_ids == pdgid
            new_ids[m] = numid
    if (new_ids==-1).sum() != 0:
        print(src_ids[new_ids==-1])
        raise Exception()
    return new_ids

class DelphesDataset(Dataset):
    def __init__(self, root, nfiles, transform=None, pre_transform=None):
        self.nfiles = nfiles
        super(DelphesDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["ev_{}.npz".format(n) for n in range(self.nfiles)]

    @property
    def processed_file_names(self):
        return ["data_{}.pt".format(n) for n in range(self.nfiles)]

    def download(self):
        pass

    def __len__(self):
        return len(self.processed_file_names)

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = load_data(raw_path)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

def load_data(fn):
    print("loading {}".format(fn))
    fi = np.load(fn)
    X = fi["X"]
    y_trk = fi["y_trk"]
    y_tower = fi["y_tower"]
    adj = fi["adj"]
    row_idx, col_idx = np.nonzero(adj)
    
    Xt = torch.Tensor(X)
    edge_index = torch.Tensor(np.stack([row_idx, col_idx], -1).T).to(dtype=torch.long)
    
    d = Data(Xt, edge_index, edge_attr=torch.ones(len(edge_index[0]), dtype=torch.float))
    d.y_trk = torch.Tensor(y_trk)
    d.y_tower = torch.Tensor(y_tower)
    new_ids_tower = encode_ids(d.y_tower[:, 0], map_candid_to_pdgid, map_candid_to_numid)
    d.y_tower[:, 0] = new_ids_tower 
    new_ids_trk = encode_ids(d.y_trk[:, 0], map_candid_to_pdgid, map_candid_to_numid)
    d.y_trk[:, 0] = new_ids_trk
    return d

class PFNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super(PFNet, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.nn = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, len(map_candid_to_numid)+3)
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, data):
        batch = data.batch
        edge_weight = data.edge_attr.squeeze(-1)

        mask_tower = data.x[:, 0] == 0
        x1 = torch.nn.functional.leaky_relu(self.conv1(data.x, data.edge_index))
        x2 = torch.nn.functional.leaky_relu(self.conv2(x1, data.edge_index))
        x3 = torch.nn.functional.leaky_relu(self.conv3(x2, data.edge_index))

        up = torch.cat([data.x, x3], axis=-1)

        cands = self.nn(up)
        cands_tower = cands[mask_tower]
        cands_tower_id = cands_tower[:, :len(map_candid_to_numid)]
        cands_tower_p = cands_tower[:, len(map_candid_to_numid):]

        cands_trk = cands[~mask_tower]
        cands_trk_id = cands_trk[:, :len(map_candid_to_numid)]
        cands_trk_p = cands_trk[:, len(map_candid_to_numid):]

        return cands_tower_id, cands_trk_id, cands_tower_p, cands_trk_p

    def encode_ids(self, ids, n=len(map_candid_to_numid)):
        ids_onehot = torch.nn.functional.one_hot(ids.to(dtype=torch.long), num_classes=n)
        return ids_onehot
    
    def decode_ids(self, ids_onehot):
        _, ids = ids_onehot.max(axis=-1)
        return ids

def to_binary(ids):
    return (ids==0).to(dtype=torch.float)

def compute_tpr_fpr(ids_true, ids_pred):
    a = to_binary(ids_true)
    b = to_binary(ids_pred)
    
    tp = float((a==1).sum())
    fp = float((a==0).sum())

    tpr = (a==1)&(b==1)
    fpr = (a==0)&(b==1)

    return float(tpr.sum())/tp, float(fpr.sum())/fp

if __name__ == "__main__":
    dataset = DelphesDataset(".", 5000)
    dataset.raw_dir = "raw2"
    dataset.processed_dir = "processed2"
    #dataset.process()

    loader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)
    model = PFNet(10, 256).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_epoch = 101
    n_train = int(0.8*len(loader))
    
    losses1 = np.zeros((n_epoch, len(loader)))
    losses2 = np.zeros((n_epoch, len(loader)))
    accuracies1 = np.zeros((n_epoch, len(loader)))
    accuracies2 = np.zeros((n_epoch, len(loader)))
   
    for i in range(n_epoch):
        t0 = time.time()
        for j, data in enumerate(loader):
            d = data.to(device=device)
            m = d.y_tower[:, 0] != 0
            d.y_tower[m, 1] = torch.log(d.y_tower[m, 1])
            m = d.y_trk[:, 0] != 0
            d.y_trk[m, 1] = torch.log(d.y_trk[m, 1])
 
            d.is_train = True
            if j >= n_train:
                d.is_train = False

            if d.is_train:
                model.train()
            else:
                model.eval()

            cands_tower_id, cands_trk_id, cands_tower_p, cands_trk_p = model(d)
            enc1 = model.encode_ids(d.y_tower[:, 0])
            dec1 = model.decode_ids(enc1)

            cands_tower_id_decoded = model.decode_ids(cands_tower_id)
            true_tower_id = d.y_tower[:, 0].to(dtype=torch.long)

            cands_trk_id_decoded = model.decode_ids(cands_trk_id)
            true_trk_id = d.y_trk[:, 0].to(dtype=torch.long)

            loss1 = 100.0*(
                torch.nn.functional.cross_entropy(cands_tower_id, true_tower_id) +
                torch.nn.functional.cross_entropy(cands_trk_id, true_trk_id)
            )
            loss2 = torch.nn.functional.mse_loss(cands_tower_p, d.y_tower[:, 1:]) + torch.nn.functional.mse_loss(cands_trk_p, d.y_trk[:, 1:])
            losses1[i, j] = float(loss1.item())
            losses2[i, j] = float(loss2.item())

            loss = loss1 + loss2           
            accuracies1[i, j]  = float((cands_tower_id_decoded==true_tower_id).sum()) / float(len(true_tower_id))
            accuracies2[i, j]  = float((cands_trk_id_decoded==true_trk_id).sum()) / float(len(true_trk_id))

            if d.is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        t1 = time.time()
        dt = t1 - t0
       
        if i%10 == 0: 
            torch.save(model.state_dict(), "model_{}.pth".format(i))
        print("epoch {} dt={:.2f} l1={:.4f}/{:.4f} l2={:.4f}/{:.4f} acc_tower={:.4f}/{:.4f} acc_track={:.4f}/{:.4f} st={:.2f}".format(
            i,
            dt,
            losses1[i, :n_train].mean(),
            losses1[i, n_train:].mean(),
            losses2[i, :n_train].mean(),
            losses2[i, n_train:].mean(),
            accuracies1[i, :n_train].mean(),
            accuracies1[i, n_train:].mean(),
            accuracies2[i, :n_train].mean(),
            accuracies2[i, n_train:].mean(),
            len(dataset)/dt
        ))
