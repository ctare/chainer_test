from chainer import datasets
import numpy as np

source_data, test_data = datasets.get_mnist()

x_data, t_data = source_data._datasets
ndata = (x_data[0] * 255).astype(np.int32).tolist()
for i in range(28):
    print(*ndata[i*28 : i*28 + 28])
