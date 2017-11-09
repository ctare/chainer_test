import ctypes
from lg import lg, blur
import os
import pylab

from chainer import datasets
import numpy as np

# source_data, test_data = datasets.get_mnist()
#
# x_data, t_data = source_data._datasets
#
# target = [None for x in range(10)]
# for index, i in enumerate(t_data):
#     if target[i] is None:
#         target[i] = (x_data[index], i)


# --- six data ---
sixdata = np.load("six.npy")

from tqdm import tqdm
import cv2
size = 50

for t, x in enumerate(sixdata):
    for n in tqdm(range(10)):
        # x = cv2.resize(x, (size, size))
        # data = blur(x, 1, size)
        data = x
        data = cv2.resize(data, (227, 227))
        np.save("data/data_{}_{}.npy".format(t, n), data)

# --- input data ---
# sixdata = []
# for i in os.listdir("pokedata"):
#     sixdata.append(np.load("pokedata/" + i))
#
# from tqdm import tqdm
# import cv2
# size = 50
#
# dec = 30
# for t, x in enumerate(sixdata):
#     x = x[:-dec, dec//2:-dec//2, :]
#     for n in tqdm(range(10)):
#         # x = cv2.resize(x, (size, size))
#         # data = blur(x, 1, size)
#         data = x
#         data = cv2.resize(data, (100, 100))
#         np.save("data/data_{}_{}.npy".format(t, n), data)
