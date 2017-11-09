import numpy as np
import chainer
from chainer import optimizers, links, functions, iterators, training

x_data = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    ], dtype=np.float32)

t_data = np.array([0, 1, 1, 0], dtype=np.int32)


class XOR(chainer.Chain):
    def __init__(self):
        in_n = 2
        out_n = 2
        super().__init__(
                l1=links.Linear(in_n, 2),
                l2=links.Linear(2, out_n)
                )

    def __call__(self, x):
        h1 = functions.sigmoid(self.l1(x))
        return self.l2(h1)


optimizer = optimizers.Adam()
xor = XOR()
model = links.Classifier(xor, lossfun=functions.softmax_cross_entropy, accfun=functions.accuracy)
# model.compute_accuracy = False
optimizer.setup(model)

x_train = chainer.datasets.TupleDataset(x_data, t_data)
train_itr = iterators.SerialIterator(x_train, batch_size=4)

updater = training.StandardUpdater(train_itr, optimizer)
trainer = training.Trainer(updater, (8000, "epoch"))

# trainer.extend(training.extensions.LogReport())
# trainer.extend(training.extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
# trainer.extend(training.extensions.ProgressBar())

trainer.run()

print("end")

d = np.array([
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        ], dtype=np.float32)

p = model.predictor(d)
print(p.data)
for i in d:
    test = np.array([i], dtype=np.float32)
    print(test)
    print(functions.softmax(xor(test)))
# print(*map(lambda x: "%.5f" % x[0].data.tolist(), xor(test)))
