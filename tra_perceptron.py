import numpy as np
import chainer
from chainer import optimizers, links, functions, iterators, training

x_data = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        ], dtype=np.float32)

t_data = np.array([
        [0],
        [1],
        [1],
        [0],
        ], dtype=np.float32)


class MLP(chainer.Chain):
    def __init__(self, in_units, n_units, out_units):
        super().__init__(
                l1=links.Linear(in_units, n_units),
                l2=links.Linear(n_units, out_units)
                )

    def __call__(self, x, t=None):
        h1 = functions.sigmoid(self.l1(x))
        return self.l2(h1)


optimizer = optimizers.Adam()
mlp = MLP(x_data.shape[1], 2, t_data.shape[1])
model = links.Classifier(mlp, lossfun=functions.mean_squared_error)
model.compute_accuracy = False
optimizer.setup(model)

x_train = chainer.datasets.TupleDataset(x_data, t_data)
train_itr = iterators.SerialIterator(x_train, batch_size=4)

updater = training.StandardUpdater(train_itr, optimizer)
trainer = training.Trainer(updater, (10000, "epoch"))

# trainer.extend(training.extensions.LogReport())
# trainer.extend(training.extensions.PrintReport(['epoch', 'main/accuracy', 'main/loss']))
# trainer.extend(training.extensions.ProgressBar())

trainer.run()

test = np.array([
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 1],
    ], dtype=np.float32)
print(test)
print(*map(lambda x: "%.5f" % x[0].data.tolist(), mlp(test)))
