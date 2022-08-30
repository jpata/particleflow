#!/use/bin/env python3

# on manivald: singularity exec /home/software/singularity/base.simg:latest python3 test/evaluate_timing.py

from graph_data import PFGraphDataset

dataset_path = "/home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi"

# Goal: measure the evaluation cost of the MLPF model as a function of input multiplicity
if __name__ == "__main__":

    full_dataset = PFGraphDataset(dataset_path)

    # events in bunches of 5
    for data_items in full_dataset:

        # loop over each event in the bunch
        for data in data_items:

            # get the input matrix
            input_matrix = data.x

            print("input_matrix.shape=", input_matrix.shape)

            # this is the number of input elements in the event
            input_multiplicity = input_matrix.shape[0]

            # task 1: plot the distribution of the input multiplicities across the events
            # using numpy.histogram and matplotlib.histogram

            # task 2: save the `data` object using:
            # torch.save(data, "data/TTbar_14TeV_TuneCUETP8M1_cfi/bin_i/file_j.pt")
            # to subfolders based on the input multiplicity binning
