#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def cuboid(ax, lower, upper):
    xl, yl, zl = lower
    xu, yu, zu = upper
    faces = np.array([
        [[[xl,xu],[xl,xu]], [[yl,yl],[yl,yl]], [[zl,zl],[zu,zu]]],
        [[[xl,xu],[xl,xu]], [[yu,yu],[yu,yu]], [[zl,zl],[zu,zu]]],
        [[[xl,xl],[xl,xl]], [[yl,yu],[yl,yu]], [[zl,zl],[zu,zu]]],
        [[[xu,xu],[xu,xu]], [[yl,yu],[yl,yu]], [[zl,zl],[zu,zu]]],
        [[[xl,xu],[xl,xu]], [[yl,yl],[yu,yu]], [[zl,zl],[zl,zl]]],
        [[[xl,xu],[xl,xu]], [[yl,yl],[yu,yu]], [[zu,zu],[zu,zu]]]
    ])
     
    for face in faces:
        ax.plot_surface(face[0], face[1], face[2], alpha=0.5, color='b')
        ax.plot_wireframe(face[0], face[1], face[2], color='k')
