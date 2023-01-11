from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="parameters/delphes-gnn-skipconn.yaml",
        help="dir containing csv files",
    )
    args = parser.parse_args()
    return args


def plot_gpu_util(df, cuda_device, ax):
    ax.plot(df["time"], df["GPU{}_util".format(cuda_device)], alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("GPU utilization [%]")
    ax.set_title("GPU{}".format(cuda_device))
    ax.grid(alpha=0.3)


def plot_gpu_power(df, cuda_device, ax):
    ax.plot(df["time"], df["GPU{}_power".format(cuda_device)], alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power consumption [W]")
    ax.set_title("GPU{}".format(cuda_device))
    ax.grid(alpha=0.3)


def plot_gpu_mem_util(df, cuda_device, ax):
    ax.plot(df["time"], df["GPU{}_mem_util".format(cuda_device)], alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("GPU memory utilization [%]")
    ax.set_title("GPU{}".format(cuda_device))
    ax.grid(alpha=0.3)


def plot_gpu_mem_used(df, cuda_device, ax):
    ax.plot(df["time"], df["GPU{}_mem_used".format(cuda_device)], alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Used GPU memory [MiB]")
    ax.set_title("GPU{}".format(cuda_device))
    ax.grid(alpha=0.3)


def plot_dfs(dfs, plot_func, suffix):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), tight_layout=True)
    for ax in axs.flat:
        ax.label_outer()

    for cuda_device, (df, ax) in enumerate(zip(dfs, axs.flat)):
        plot_func(df, cuda_device, ax)
    plt.suptitle("{}".format(file.stem))
    plt.savefig(args.dir + "/{}_{}.jpg".format(file.stem, suffix))


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
            dfs.append(
                pd.DataFrame(
                    {
                        "GPU{}_util".format(ii): df[df[" pci.bus_id"] == gpu][" utilization.gpu [%]"].map(
                            lambda x: int(x.split(" ")[1])
                        ),
                        "GPU{}_power".format(ii): df[df[" pci.bus_id"] == gpu][" power.draw [W]"].map(
                            lambda x: float(x.split(" ")[1])
                        ),
                        "GPU{}_mem_util".format(ii): df[df[" pci.bus_id"] == gpu][" utilization.memory [%]"].map(
                            lambda x: int(x.split(" ")[1])
                        ),
                        "GPU{}_mem_used".format(ii): df[df[" pci.bus_id"] == gpu][" memory.used [MiB]"].map(
                            lambda x: int(x.split(" ")[1])
                        ),
                        "time": df[df[" pci.bus_id"] == gpu]["timestamp"].map(
                            lambda x: datetime.strptime(x, "%Y/%m/%d %H:%M:%S.%f").timestamp() - start_t
                        ),
                    }
                ).dropna()
            )

        plot_dfs(dfs, plot_gpu_util, "gpu_util")
        plot_dfs(dfs, plot_gpu_power, "gpu_power")
        plot_dfs(dfs, plot_gpu_mem_used, "gpu_mem_used")
        plot_dfs(dfs, plot_gpu_mem_util, "gpu_mem_util")
