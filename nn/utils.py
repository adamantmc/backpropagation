import os
import shutil
from tabulate import tabulate
from cProfile import Profile
import pstats
import matplotlib.pyplot as plt


def plot_losses(training_losses, validation_losses, savepath="./losses.png"):
    fig = plt.figure()
    plt.plot(validation_losses, label="Validation set loss")
    plt.plot(training_losses, label="Training set loss")

    plt.legend()
    plt.xlabel("Epochs")

    plt.savefig(savepath)


def plot_histograms(self, data):
    fig, axs = plt.subplots(len(data))
    for index, k in enumerate(data):
        if len(data) == 1:
            ax = axs
        else:
            ax = axs[index]

        ax.hist(data[k])
        ax.set_label(k)

    plt.show()


def dump_values(x, a, z, w, dw, db, epoch):
    dir = "./epoch_{}_values".format(epoch)

    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)

    os.makedirs(dir)
    data = {"a": a, "z": z, "w": w, "dw": dw, "db": db}

    for d in data:
        vals = data[d]

        for index, arr in enumerate(vals):
            with open(os.path.join(dir, "{}{}".format(d, index+1)), "w") as f:
                table = tabulate(arr)
                f.write(table + "\n")

    with open(os.path.join(dir,"x"), "w") as f:
        table = tabulate(x)
        f.write(table + "\n")


def profile(sort_args=['cumulative'], print_args=[10]):
    profiler = Profile()

    def decorator(fn):
        def inner(*args, **kwargs):
            result = None
            try:
                result = profiler.runcall(fn, *args, **kwargs)
            finally:
                stats = pstats.Stats(profiler)
                stats.strip_dirs().sort_stats(*sort_args).print_stats(*print_args)
            return result
        return inner
    return decorator