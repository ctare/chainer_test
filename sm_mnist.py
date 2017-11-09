import matplotlib.pyplot as plt
import numpy as np
from chainer import datasets
from chainer import cuda, Variable, optimizers, Chain, training, iterators, serializers
import chainer
import chainer.functions  as F
import chainer.links as L
import sys

can_use_gpu = False
try:
    xp = cuda.cupy
    can_use_gpu = True
except:
    xp = np

batchsize = 100
n_epoch = 20
n_units = 1000
output_n = 10
 
source_data, test_data = datasets.get_mnist()

x_data, t_data = source_data._datasets
N, img_size = x_data.shape

x_test, t_test = test_data._datasets
N_test, img_size_test = x_test.shape

class MNIST(Chain):
    def __init__(self):
        super().__init__(
                l1=L.Linear(img_size, n_units),
                l2=L.Linear(n_units, n_units),
                l3=L.Linear(n_units, output_n))

    def __call__(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        h1 = F.dropout(F.relu(self.l2(h1)))
        return self.l3(h1)


optimizer = optimizers.Adam()
mnist_model = MNIST()
model = L.Classifier(mnist_model, lossfun=F.softmax_cross_entropy, accfun=F.accuracy)
# model.compute_accuracy = False
optimizer.setup(model)

if can_use_gpu:
    gpu_device = 0
    cuda.get_device(gpu_device).use()
    model.to_gpu(gpu_device)

lgdata = xp.load("lgdata.npy").astype(xp.float32)
lgtarget = xp.load("lglabel.npy").astype(xp.int32)

train_itr = iterators.SerialIterator(chainer.datasets.TupleDataset(lgdata, lgtarget), batch_size=batchsize)
# train_itr = iterators.SerialIterator(source_data, batch_size=batchsize)
updater = training.StandardUpdater(train_itr, optimizer)
trainer = training.Trainer(updater, (n_epoch, "epoch"))

trainer.extend(training.extensions.LogReport())
trainer.extend(training.extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
trainer.extend(training.extensions.ProgressBar())

import os
model_path = "my_mnist.h5"
if os.path.exists(model_path):
    serializers.load_hdf5(model_path, model)
    p = F.softmax(model.predictor(x_test))
    print(x_test.shape)
    print(sum(xp.argmax(p.data, axis=1) == t_test) / len(t_test))
else:
    trainer.run()
    mnist_model.to_cpu()
    serializers.save_hdf5(model_path, model)
