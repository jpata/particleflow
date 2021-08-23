from glob import glob
import sys, os

import os.path as osp
import pickle as pkl
import math, time, tqdm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

#Check if the GPU configuration has been provided
import torch
use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
        if multi_gpu:
            print('Will use multi_gpu..')
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        else:
            print('Will use single_gpu..')
except Exception as e:
    print("Could not import setGPU, running CPU-only")

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
    print("GPU model:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

import torch_geometric

import pytorch_delphes
import plotting

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

#Get a unique directory name for the model
def get_model_fname(dataset, model, n_train, n_epochs, lr, target_type, batch_size, alpha, task, title):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_ntrain_{}_nepochs_{}_batch_size_{}_lr_{}_alpha_{}_{}_{}'.format(
        model_name,
        target_type,
        n_train,
        n_epochs,
        batch_size,
        lr,
        alpha,
        task,
        title)
    return model_fname


if __name__ == "__main__":

    args = pytorch_delphes.parse_args()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'train': True, 'n_train': 1, 'n_valid': 1, 'n_test': 1, 'n_epochs': 2, 'patience': 100, 'hidden_dim': 256, 'hidden_dim_nn1': 64, 'input_encoding': 12, 'encoding_dim': 64,
    # 'batch_size': 1, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../test_tmp_delphes/data/pythia8_qcd',
    # 'outpath': '../prp/models/yee/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 2e-4,
    # 'space_dim': 4, 'propagate_dimensions': 22, 'nearest': 16, 'overwrite': True,
    # 'load': False, 'load_epoch': 1, 'load_model': 'PFNet7_gen_ntrain_1_nepochs_2_batch_size_1_lr_0.001_alpha_0.0002_both_noembeddingsnoskip_nn1_nn3',
    # 'classification_only': False, 'nn1': True, 'nn3': True, 'encoding_of_clusters': False, 'embedding_dim': 0, 'nn0track': False, 'nn0cluster': False, 'title': 'noembeddings',
    # 'make_predictions_train': False, 'make_plots_train': False, 'make_predictions_valid': False, 'make_plots_valid': False, 'make_predictions_test': True, 'make_plots_test': True,
    # 'optimized': False})

    # define the dataset (assumes the data exists as .pt files in "processed")
    print('Processing the data..')
    full_dataset_ttbar = pytorch_delphes.PFGraphDataset(args.dataset)
    full_dataset_qcd = pytorch_delphes.PFGraphDataset(args.dataset_qcd)

    # constructs a loader from the data to iterate over batches
    print('Constructing data loaders..')
    train_loader, valid_loader = pytorch_delphes.data_to_loader_ttbar(full_dataset_ttbar, args.n_train, args.n_valid, batch_size=args.batch_size)
    test_loader = pytorch_delphes.data_to_loader_qcd(full_dataset_qcd, args.n_test, batch_size=args.batch_size)

    # element parameters
    input_dim = 12

    #one-hot particle ID and momentum
    output_dim_id = 6
    output_dim_p4 = 6

    if args.optimized:
        model_classes = {"PFNet7": pytorch_delphes.PFNet7_opt}
    else:
        model_classes = {"PFNet7": pytorch_delphes.PFNet7}

    model_class = model_classes[args.model]
    model_kwargs = {'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'hidden_dim_nn1': args.hidden_dim_nn1,
                    'input_encoding': args.input_encoding,
                    'encoding_dim': args.encoding_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4,
                    'space_dim': args.space_dim,
                    'propagate_dimensions': args.propagate_dimensions,
                    'nearest': args.nearest,
                    'target': args.target,
                    'nn1': args.nn1,
                    'nn3': args.nn3}

    if args.load:
            print('Loading a previously trained model..')
            model = model_class(**model_kwargs)
            outpath = args.outpath + args.load_model
            PATH = outpath + '/epoch_' + str(args.load_epoch) + '_weights.pth'

            state_dict = torch.load(PATH, map_location=device)

            if "DataParallel" in args.load_model:   # if the model was trained using DataParallel then we do this
                state_dict = torch.load(PATH, map_location=device)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove module.
                    new_state_dict[name] = v
                    # print('name is:', name)
                state_dict=new_state_dict

            model.load_state_dict(state_dict)

            if multi_gpu:
                model = torch_geometric.nn.DataParallel(model)
                #model = torch.nn.parallel.DistributedDataParallel(model)    ### TODO: make it compatible with DDP

            model.to(device)

            if args.train:
                print("Training a previously trained model..")

    elif args.train:
        #instantiate the model
        print('Instantiating a model..')
        model = model_class(**model_kwargs)

        if multi_gpu:
            print("Parallelizing the training..")
            model = torch_geometric.nn.DataParallel(model)
            #model = torch.nn.parallel.DistributedDataParallel(model)    ### TODO: make it compatible with DDP

        model.to(device)

    if args.train:
        args.title=args.title+'noskip'
        if args.nn1:
            args.title=args.title+'_nn1'
        if args.nn3:
            args.title=args.title+'_nn3'

        if args.classification_only:
            model_fname = get_model_fname(args.dataset, model, args.n_train, args.n_epochs, args.lr, args.target, args.batch_size, args.alpha, "clf", args.title)
        else:
            model_fname = get_model_fname(args.dataset, model, args.n_train, args.n_epochs, args.lr, args.target, args.batch_size,  args.alpha, "both", args.title)

        outpath = osp.join(args.outpath, model_fname)
        if osp.isdir(outpath):
            if args.overwrite:
                print("model output {} already exists, deleting it".format(outpath))
                import shutil
                shutil.rmtree(outpath)
            else:
                print("model output {} already exists, please delete it".format(outpath))
                sys.exit(0)
        try:
            os.makedirs(outpath)
        except Exception as e:
            pass

        with open('{}/model_kwargs.pkl'.format(outpath), 'wb') as f:
            pkl.dump(model_kwargs, f,  protocol=pkl.HIGHEST_PROTOCOL)

        if not os.path.exists(outpath + '/confusion_matrix_plots/'):
            os.makedirs(outpath + '/confusion_matrix_plots/')

        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print(model)
        print(model_fname)

        model.train()
        pytorch_delphes.train_loop(model, device, multi_gpu,
                                   train_loader, valid_loader, test_loader,
                                   args.n_epochs, args.patience, optimizer, args.alpha, args.target,
                                   output_dim_id, args.classification_only, outpath)

    model.eval()

    # evaluate on training data..
    if not osp.isdir(outpath+'/train_loader'):
        os.makedirs(outpath+'/train_loader')
    if not osp.isdir(outpath+'/train_loader/resolution_plots'):
        os.makedirs(outpath+'/train_loader/resolution_plots')
    if not osp.isdir(outpath+'/train_loader/distribution_plots'):
        os.makedirs(outpath+'/train_loader/distribution_plots')
    if not osp.isdir(outpath+'/train_loader/multiplicity_plots'):
        os.makedirs(outpath+'/train_loader/multiplicity_plots')
    if not osp.isdir(outpath+'/train_loader/efficiency_plots'):
        os.makedirs(outpath+'/train_loader/efficiency_plots')

    if args.make_predictions_train:
        pytorch_delphes.make_predictions(model, multi_gpu, train_loader, outpath+'/train_loader', args.target, device, args.n_epochs, which_data="training data")
    if args.make_plots_train:
        plotting.make_plots(model, train_loader, outpath+'/train_loader', args.target, device, args.n_epochs, which_data="training data")

    # evaluate on validation data..
    if not osp.isdir(outpath+'/valid_loader'):
        os.makedirs(outpath+'/valid_loader')
    if not osp.isdir(outpath+'/valid_loader/resolution_plots'):
        os.makedirs(outpath+'/valid_loader/resolution_plots')
    if not osp.isdir(outpath+'/valid_loader/distribution_plots'):
        os.makedirs(outpath+'/valid_loader/distribution_plots')
    if not osp.isdir(outpath+'/valid_loader/multiplicity_plots'):
        os.makedirs(outpath+'/valid_loader/multiplicity_plots')
    if not osp.isdir(outpath+'/valid_loader/efficiency_plots'):
        os.makedirs(outpath+'/valid_loader/efficiency_plots')

    if args.make_predictions_valid:
        pytorch_delphes.make_predictions(model, multi_gpu, valid_loader, outpath+'/valid_loader', args.target, device, args.n_epochs, which_data="validation data")
    if args.make_plots_valid:
        plotting.make_plots(model, valid_loader, outpath+'/valid_loader', args.target, device, args.n_epochs, which_data="validation data")

    # evaluate on testing data..
    if not osp.isdir(outpath+'/test_loader'):
        os.makedirs(outpath+'/test_loader')
    if not osp.isdir(outpath+'/test_loader/resolution_plots'):
        os.makedirs(outpath+'/test_loader/resolution_plots')
    if not osp.isdir(outpath+'/test_loader/distribution_plots'):
        os.makedirs(outpath+'/test_loader/distribution_plots')
    if not osp.isdir(outpath+'/test_loader/multiplicity_plots'):
        os.makedirs(outpath+'/test_loader/multiplicity_plots')
    if not osp.isdir(outpath+'/test_loader/efficiency_plots'):
        os.makedirs(outpath+'/test_loader/efficiency_plots')

    if args.make_predictions_test:
        if args.load:
            pytorch_delphes.make_predictions(model, multi_gpu, test_loader, outpath+'/test_loader', args.target, device, args.load_epoch, which_data="testing data")
        else:
            pytorch_delphes.make_predictions(model, multi_gpu, test_loader, outpath+'/test_loader', args.target, device, args.n_epochs, which_data="testing data")
    if args.make_plots_test:
        if args.load:
            plotting.make_plots(model, test_loader, outpath+'/test_loader', args.target, device, args.load_epoch, which_data="testing data")
        else:
            plotting.make_plots(model, test_loader, outpath+'/test_loader', args.target, device, args.n_epochs, which_data="testing data")


## -----------------------------------------------------------
# to retrieve a stored variable in pkl file
# import pickle as pkl
# with open('../../test_tmp_delphes/experiments/PFNet7_gen_ntrain_2_nepochs_3_batch_size_3_lr_0.0001/confusion_matrix_plots/cmT_normed_epoch_0.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     a = pkl.load(f)
#
# with open('../../data/pythia8_qcd/raw/tev14_pythia8_qcd_10_0.pkl', 'rb') as pickle_file:
#     data = pkl.load(pickle_file)
#
# data.keys()
