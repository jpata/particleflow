import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import time

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="parameters/delphes-gnn-skipconn.yaml", help="dir containing csv files")
    args = parser.parse_args()
    return args


def plot_gpu_util(df, cuda_device):
    plt.figure(figsize=(12,9))
    plt.plot(df["time"], df["GPU{}_util".format(cuda_device)], alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("GPU utilization [%]")
    plt.title("GPU{}".format(cuda_device))
    plt.grid(alpha=0.3)


def plot_gpu_util(df, cuda_device, ax):
    ax.plot(df["time"], df["GPU{}_util".format(cuda_device)], alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("GPU utilization [%]")
    ax.set_title("GPU{}".format(cuda_device))
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    args = parse_args()
    csv_files = list(Path(args.dir).glob("*.csv"))

    for file in csv_files:
        print(file)
        df = pd.read_csv(str(file))
        start_time = df["timestamp"].iloc[0]
        start_t = datetime.strptime(start_time, "%Y/%m/%d %H:%M:%S.%f").timestamp()
        dfs = []
        for ii, gpu in enumerate(np.unique(df[" pci.bus_id"].values)):
            dfs.append(pd.DataFrame({
                "GPU{}_util".format(ii): df[df[" pci.bus_id"] == gpu][" utilization.gpu [%]"].map(lambda x: int(x.split(" ")[1])),
                "time": df[df[" pci.bus_id"] == gpu]["timestamp"].map(lambda x: datetime.strptime(x, "%Y/%m/%d %H:%M:%S.%f").timestamp() - start_t),
            }).dropna())
    
        fig, axs = plt.subplots(2, 2, figsize=(12,9), tight_layout=True)
        for ax in axs.flat:
            ax.label_outer()

        for cuda_device, (df, ax) in enumerate(zip(dfs, axs.flat)):
            plot_gpu_util(df, cuda_device, ax)
        plt.suptitle("{}".format(file.stem))
        plt.savefig(args.dir + "/{}_gpu_util.jpg".format(file.stem))