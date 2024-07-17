import json
from pathlib import Path

import click
import matplotlib.pyplot as plt


@click.group()
@click.help_option("-h", "--help")
def main():
    pass


@main.command()
@click.help_option("-h", "--help")
@click.option(
    "-p",
    "--path",
    help="path to json file or dir containing json files",
    type=click.Path(),
)
@click.option("-y", "--ylabel", default=None, help="Y-axis label", type=str)
@click.option("-x", "--xlabel", default="Step", help="X-axis label", type=str)
@click.option("-t", "--title", default=None, help="X-axis label", type=str)
@click.option(
    "-s",
    "--save_dir",
    default=None,
    help="X-axis label",
    type=click.Path(),
)
def plot_cometml_json(path, ylabel, xlabel, title=None, save_dir=None):
    path = Path(path)

    if path.is_dir():
        json_files = path.glob("*.json")
    else:
        json_files = [path]

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        plt.figure(figsize=(12, 6))

        for ii, metric in enumerate(data):
            # Figure out val and train relationship and ordering in data list
            if "val" in metric["name"]:
                pass
            else:
                try:
                    val_metric = data[ii + 1]
                except IndexError:
                    val_metric = data[ii - 1]
                if ("val_" + metric["name"]) != val_metric["name"]:
                    val_metric = data[ii - 1]
                    if ("val_" + metric["name"]) != val_metric["name"]:
                        raise ValueError("The val and train metrics don't match, {}, {}".format("val_" + metric["name"], val_metric["name"]))

                pp = plt.plot(
                    metric["x"],
                    metric["y"],
                    label=metric["name"],
                    linestyle="-",
                )
                color = pp[0].get_color()
                plt.plot(
                    val_metric["x"],
                    val_metric["y"],
                    label=val_metric["name"],
                    linestyle="--",
                    color=color,
                )

        plt.legend()
        plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title("")
        if save_dir:
            plt.savefig(str(Path(save_dir) / (json_file.stem + ".jpg")))
    if not save_dir:
        plt.show()


if __name__ == "__main__":
    main()
