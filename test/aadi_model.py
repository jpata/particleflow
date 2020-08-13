import pickle
import time
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels

from train_end2end import PFNet7



#loading kwargs pickle
pickle_in = open("../../model_init_data/model_kwargs.pkl","rb")
kwargs_dict = pickle.load(pickle_in)

#binned data 
binned = "/home/aadi/praktika/binned_data/"#processed

#loading predetermined weights
device = torch.device('cpu')
predet_weights = torch.load("../../model_init_data/weights.pth",map_location = device)
#6k_bin at the moment
bin_data = PFGraphDataset(binned)
print(len(bin_data))




#loading model
model = PFNet7(**kwargs_dict)
missing_key_fix = {k.replace("module.", ""): v for k, v in predet_weights.items()}
model.load_state_dict(missing_key_fix)
model.eval()




lst = []
for i in range(10):
    if i <10 :
        x = bin_data[i]
        start_time= time.time()
        model(x)
        print("---%s seconds " % (time.time() -start_time))
        lst.append(time.time() -start_time)
print(np.mean(lst))





#start_time= time.time()
#model(bin_data[0])
#print("---%s seconds " % (time.time() -start_time))




