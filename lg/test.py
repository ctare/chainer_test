#coding: utf-8
import numpy as np
import pylab
import os
position = 0
for index, i in enumerate(os.listdir("data")[position:position + 6]):
    data = np.load("data/" + i)

    pylab.subplot(5, 5, index + 1)
    pylab.axis('off')
    pylab.imshow(data)
    pylab.title(i)
pylab.show()
