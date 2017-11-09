import os
import numpy as np
from tqdm import tqdm

data = []
label = []
for i in tqdm(os.listdir("./data")):
    target = int(i[5])
    d = np.load("./data/" + i)
    data.append(d)
    label.append(target)

import sys
if len(sys.argv) > 1:
    print("saved xtdata/{}".format(sys.argv[1]))
    np.save("xtdata/x_{}.npy".format(sys.argv[1]), np.array(data))
    np.save("xtdata/t_{}.npy".format(sys.argv[1]), np.array(label))
    print("end")
else:
    print("please filename")
