import numpy as np
import chainer
from chainer import optimizers
import chainer.links as L
import chainer.functions as F

source = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        ]

target = [
        [0],
        [1],
        [1],
        [0],
        ]

dataset = {}
dataset["source"] = np.array(source, dtype=np.float32)
dataset["target"] = np.array(target, dtype=np.float32)

N = len(source)

in_units = 2
n_units = 2
out_units = 1

model = chainer.Chain(
        l1=L.Linear(in_units, n_units),
        l2=L.Linear(n_units, out_units),
        )


def forward(x, t):
    h1 = F.sigmoid(model.l1(x))
    return model.l2(h1)


# learn

optimizer = optimizers.Adam()
optimizer.setup(model)

loss_val = 100
epoch = 0
n_epoch = int(input("epoch n: "))
while loss_val > 1e-5 and epoch < n_epoch:
    x = chainer.Variable(np.asarray(dataset["source"]))
    t = chainer.Variable(np.asarray(dataset["target"]))

    model.zerograds()
    y = forward(x, t)
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()

    if not epoch % 1000:
        loss_val = loss.data
        print("epoch:", epoch)
        print("x:\n", x.data)
        print("t:\n", t.data)
        print("y:")
        for v in y.data:
            print(*map(lambda x: "%.5f" % x, v))
        print("train mean loss = %.5f" % loss_val)
        print("----------")
    epoch += 1


