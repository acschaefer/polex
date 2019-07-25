#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch


def _take(data, index):
    slice = data
    for i in range(len(index)):
        slice = np.take(slice, min(slice.shape[0]-1, index[i]), axis=0)

    return slice


def matshow(data, matnames=[], dimnames=[]):
    if not isinstance(data, list):
        data = [data]

    for i in range(len(data)):
        if type(data[i]) is torch.Tensor:
            data[i] = data[i].numpy()

    ndim = max([d.ndim for d in data])

    for i in range(len(data)):
        while data[i].ndim < ndim:
            data[i] = np.expand_dims(data[i], axis=0)

    shape = []
    for dim in range(ndim - 2):
        shape.append(max(d.shape[dim] for d in data))

    figure, axes = plt.subplots(
        1, len(data), sharex=True, sharey=True, squeeze=False)

    for i in range(len(data)):
        axes[0,i].imshow(_take(data[i], [0] * (ndim-2)), 
            vmin=np.amin(data[i]), vmax=np.amax(data[i]),
            interpolation=None, origin='lower',
            extent=[0.0, data[i].shape[-1], 0.0, data[i].shape[-2]])

    for i in range(min(len(data), len(matnames))):
        axes[0,i].set_title(matnames[i])

    sliders = []
    updatefuncs = []
    bottom = np.linspace(0.0, 0.1, ndim)[1:-1]
    for i in range(len(shape)):
        sliderax = plt.axes([0.2, bottom[i], 0.6, 0.02], 
            facecolor='lightgoldenrodyellow')

        if i < len(dimnames):
            label = dimnames[i]
        else:
            label = 'Axis {}'.format(i)
        sliders.append(Slider(sliderax, label=label, 
            valmin=0, valmax=shape[i]-1, valinit=0, valstep=1))

        def update(val):
            indices = [int(slider.val) for slider in sliders]
            for j in range(axes.size):
                axes[0,j].images[0].set_array(_take(data[j], indices))
            figure.canvas.draw_idle()
        updatefuncs.append(update)

        sliders[i].on_changed(updatefuncs[i])

    plt.show()


if __name__ == '__main__':
    matshow(np.random.rand(100, 15, 25, 64, 64))
    matshow(np.random.rand(100, 15, 25, 64, 64), 
        matnames=['Matrix A', 'Matrix B'], 
        dimnames=['depth', 'height', 'width'])
    matshow([np.random.rand(64, 64, 64), np.random.rand(10, 10, 64)])
    matshow([np.random.rand(9, 11), np.random.rand(12, 12, 6)], 
        matnames=['9x11', '12x12x6'])
    matshow(np.random.rand(100, 100, 100))
