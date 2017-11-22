import numpy as np
import chainer
from chainer import links as L, functions as F
import pylab
import cv2

x_data = np.load("lg/xtdata/x_pokedata1000.npy").astype(np.float32) / 255
t_data = np.load("lg/xtdata/t_pokedata1000.npy").astype(np.int32)

# dec = 5
# x_data = x_data[::dec]
# t_data = t_data[::dec]
#
x_data = x_data.transpose(0, 3, 1, 2)


class IMAGE(chainer.Chain):
    def __init__(self):
        super().__init__(
                # (227 - 11) / 4 + 1 => 55
                c1=L.Convolution2D(3, 96, 11, stride=4),
                # pooling 3, stride=2 | 27
                b1=L.BatchNormalization(96),
                # (27 + 2*2 - 5) / 1 => 27
                c2=L.Convolution2D(96, 256, 5, pad=2),
                # pooling 3, stride=2 | 13
                b2=L.BatchNormalization(256),
                # (13 + 1*2 - 3) / 1 => 13
                c3=L.Convolution2D(256, 384, 3, pad=1),
                # (13 + 1*2 - 3) / 1 => 13
                c4=L.Convolution2D(384, 384, 3, pad=1),
                # (13 + 1*2 - 3) / 1 => 13
                c5=L.Convolution2D(384, 256, 3, pad=1),
                # pooling 3, stide=2 | 6
                l1=L.Linear(256 * 6 * 6, 4096),
                l2=L.Linear(4096, 1024),
                l3=L.Linear(1024, 6),
                )

    def show(self, h, size):
        print(h.data.shape)
        img = h.data[0]
        ch = len(img)
        for i, v in enumerate(img, 1):
            pylab.subplot(int(ch ** 0.5) + 1, int(ch ** 0.5) + 1, i)
            pylab.axis('off')
            pylab.imshow(v)

    def __call__(self, x):
        h = x
        h = F.max_pooling_2d(F.relu(self.b1(self.c1(h))), 3, stride=2)
        self.show(h, 27)
        h = F.max_pooling_2d(F.relu(self.b2(self.c2(h))), 3, stride=2)
        # h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.c1(h))), 3, stride=2)
        # h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.c2(h))), 3, stride=2)
        h = F.relu(self.c3(h))
        h = F.relu(self.c4(h))
        h = F.max_pooling_2d(F.relu(self.c5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.l1(h)))
        h = F.dropout(F.relu(self.l2(h)))
        return self.l3(h)


optimizer = chainer.optimizers.Adam()
imodel = IMAGE()
model = L.Classifier(imodel, lossfun=F.softmax_cross_entropy, accfun=F.accuracy)
optimizer.setup(model)

import os
model_path = "poke.h5"
cnt = 0
if os.path.exists(model_path):
    chainer.serializers.load_hdf5(model_path, model)
    x_data = np.load("sixdata227.npy")

    #  - - lea im - -
    x_tests = np.load("lg/xtdata/x_poketestLG.npy")
    #
    # x_gyara = x_tests[1 * 10][:, :, ::-1]
    # x_gyara = np.array([cv2.resize(x_gyara, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)

    x_kaguya = x_tests[0 * 10][:, :, ::-1]
    x_kaguya = np.array([cv2.resize(x_kaguya, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)

    # x_pori = x_tests[5 * 10][:, :, ::-1]
    # x_pori = np.array([cv2.resize(x_pori, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)
    #
    # x_gard = x_tests[2 * 10][:, :, ::-1]
    # x_gard = np.array([cv2.resize(x_gard, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)
    # imodel(x_gyara)
    # cnt += 1
    imodel(x_kaguya)
    # cnt += 1
    # imodel(x_pori)
    # cnt += 1
    # imodel(x_gard)
    pylab.show()
    # - - - - - 
    
    # for i in range(6):
    #     # x_test = np.r_[np.zeros((25, 100, 3), dtype=np.uint8), x_test[:-25, :, ::-1]]
    #     for idx, x_test in enumerate(x_tests[i*10: i*10 + 10]):
    #         x_test = x_test[:, :, :]
    #         x_tmp = x_test
    #
    #         x_test = np.array([cv2.resize(x_test, (227, 227))], dtype=np.float32).transpose(0, 3, 1, 2)
    #
    #         # x_data = np.load("sixdata227.npy").astype(np.float32) / 255
    #         # t_data = np.load("sixlabel.npy").astype(np.int32)
    #         #
    #         # x_data = x_data.transpose(0, 3, 1, 2)
    #         #
    #         # print(x_data.shape)
    #         p = model.predictor(x_test)
    #         # print(p)
    #         # print("---")
    #         # print(p.shape)
    #         # print("===")
    #         p = F.softmax(p)
    #         # print(p)
    #         print(np.argmax(p.data))
    #
    #         pylab.subplot(14, 10, idx + 20*i + 1 )
    #         pylab.axis('off')
    #         pylab.imshow(x_tmp.astype(np.uint8))
    #         pylab.title("%d" % np.argmax(p.data))
    # for s in range(6):
    #     pylab.subplot(14, 10, 120 + s + 1)
    #     pylab.axis('off')
    #     pylab.imshow(x_data[200 * s + 1])
    #     pylab.title(s)
    # print("===")
    # pylab.show()
    # print(x_test.shape)
    # print(sum(np.argmax(p.data, axis=1) == t_test) / len(t_test))
else:
    x_train = chainer.datasets.TupleDataset(x_data, t_data)
    train_itr = chainer.iterators.SerialIterator(x_train, batch_size=240)

    updater = chainer.training.StandardUpdater(train_itr, optimizer)
    trainer = chainer.training.Trainer(updater, (2, "epoch"))

    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(["epoch", "main/loss", "main/accuracy"]))
    trainer.extend(chainer.training.extensions.ProgressBar())

    trainer.run()
    chainer.serializers.save_hdf5(model_path, model)

# c1 = L.Convolution2D(3, 96, 12, stride=4)
# h = c1(x_data, dtype=np.float32))
# print(h.shape)
# h = F.max_pooling_2d(h, 3, stride=2)
# print(h.shape)
# c2 = L.Convolution2D(96, 256, 5, pad=2)
# h = c2(h)
# print(h.shape)
# h = F.max_pooling_2d(h, 3, stride=2)
# print(h.shape)
# c3 = L.Convolution2D(256, 384, 3, pad=1)
# h = c3(h)
# print(h.shape)
# c4 = L.Convolution2D(384, 384, 3, pad=1)
# h = c4(h)
# print(h.shape)
# c5 = L.Convolution2D(384, 256, 3, pad=1)
# h = c5(h)
# print(h.shape)
# h = F.max_pooling_2d(h, 3, stride=2)
# print(h.shape)
# l1 = L.Linear(256 * 6 * 6, 4096)
# h = l1(h)
# print(h.shape)
# l2 = L.Linear(4096, 1024)
# h = l2(h)
# print(h.shape)
# l3 = L.Linear(1024, 6)
# h = l3(h)
# print(h.shape)
# print(F.softmax(h))
