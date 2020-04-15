import os
import shutil
from tabulate import tabulate


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