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

pdgid_to_numid = {
    0: 0,
    22: 1,
    11: 2,
    -11: 3,
    211: 4,
    -211: 5,
    130: 6,
}

def encode_ids(src_ids, encoding_map):
    new_ids = torch.zeros_like(src_ids)
    placeholder = max(encoding_map.values()) + 1
    new_ids[:] = placeholder
    for k, v in encoding_map.items():
        m = src_ids == k
        new_ids[m] = v
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
    y = fi["y"]
    adj = fi["adj"]
    row_idx, col_idx = np.nonzero(adj)
    
    Xt = torch.Tensor(X)
    yt = torch.Tensor(y).to(dtype=torch.float)
    edge_index = torch.Tensor(np.stack([row_idx, col_idx], -1).T).to(dtype=torch.long)
    
    d = Data(Xt, edge_index, edge_attr=torch.ones(len(edge_index[0]), dtype=torch.float))
    d.y_id = encode_ids(yt[:, 0], pdgid_to_numid).to(dtype=torch.float).unsqueeze(-1)
    d.y_p = yt[:, 1:]
    print(d)
    return d

class PFNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super(PFNet, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False)
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, len(pdgid_to_numid) + 1),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, data):
        batch = data.batch
        edge_weight = data.edge_attr.squeeze(-1)

        #Run a second convolution with the new edges
        x = torch.nn.functional.leaky_relu(self.conv1(data.x, data.edge_index))

        up = torch.cat([data.x, x], axis=-1)
        cand_id = self.nn1(up)
        cand_p = self.nn2(up)
        return cand_id, cand_p

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

    loader = DataLoader(dataset, batch_size=2, pin_memory=True, shuffle=False)
    model = PFNet(6, 512).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_epoch = 200
    n_train = int(0.8*len(loader))
    
    losses1 = np.zeros((n_epoch, len(loader)))
    losses2 = np.zeros((n_epoch, len(loader)))
    accuracies = np.zeros((n_epoch, len(loader)))
   
    for i in range(n_epoch):
        t0 = time.time()
        for j, data in enumerate(loader):
            d = data.to(device=device)
            d.x = d.x[:, :6]
            d.is_train = True
            if j >= n_train:
                d.is_train = False

            pred_id_onehot, pred_p = model(d)
            _, pred_id_num = torch.max(pred_id_onehot, -1)
            #pred_p[pred_id_num==0, :] *= 0.00001
            loss1 = 100*torch.nn.functional.cross_entropy(pred_id_onehot, d.y_id[:, 0].to(dtype=torch.long))
            loss2 = torch.nn.functional.mse_loss(pred_p, d.y_p)
            loss = loss1 + loss2
            
 
            losses1[i, j] = float(loss1.item())
            losses2[i, j] = float(loss2.item())
            acc = (pred_id_num==d.y_id[:, 0]).sum() / float(len(d.y_id))
            accuracies[i, j] = acc
            
            if d.is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        t1 = time.time()
        dt = t1 - t0
       
        if i%10 == 0: 
            torch.save(model.state_dict(), "model_{}.pth".format(i))
        print("epoch {} dt={:.2f} l1={:.4f}/{:.4f} l2={:.4f}/{:.4f} acc={:.4f}/{:.4f} st={:.2f}".format(
            i,
            dt,
            losses1[i, :n_train].mean(),
            losses1[i, n_train:].mean(),
            losses2[i, :n_train].mean(),
            losses2[i, n_train:].mean(),
            accuracies[i, :n_train].mean(),
            accuracies[i, n_train:].mean(),
            len(dataset)/dt
        ))
