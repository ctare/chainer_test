import numpy as np
import chainer
from chainer import optimizers, links, functions, iterators, training

source = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        ], dtype=np.float32)

target = np.array([
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
mlp = MLP(2, 2, 1)
accfun = lambda x, t: functions.sum(1 - abs(x-t))/x.size
model = links.Classifier(mlp, lossfun=functions.mean_squared_error, accfun=accfun)
optimizer.setup(model)

x_train = chainer.datasets.TupleDataset(source, target)
train_itr = iterators.SerialIterator(x_train, batch_size=4)

updater = training.StandardUpdater(train_itr, optimizer)
trainer = training.Trainer(updater, (8000, "epoch"), out="result")

# trainer.extend(training.extensions.LogReport())
# trainer.extend(training.extensions.PrintReport(['epoch', 'main/accuracy', 'main/loss']))
# trainer.extend(training.extensions.ProgressBar())

trainer.run()

print("end")

test = np.array([
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 1],
    ], dtype=np.float32)
print(test)
print(*map(lambda x: "%.5f" % x[0].data.tolist(), mlp(test)))
