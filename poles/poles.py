#!/usr/bin/env python

import numpy as np
import scipy.stats
import skimage.feature
import skimage.measure
import torch


device = torch.device('cuda')
polesides = range(1, 5+1)
minscore = 0.6
minheight = 1.0
freelength = 0.2
dstop = 1.0e-3
normstd = 0.2


def detect_poles(occupancymap, mapsize):
    f = int(np.round(freelength / mapsize[0]))
    polemapshape = occupancymap.shape - np.array([2*f, 2*f, 0])
    
    ogm = torch.tensor(
        occupancymap, device=device).permute([2, 0, 1]).unsqueeze(1)
    accuscores = torch.zeros(
        tuple(np.hstack([len(polesides), polemapshape[[2, 0, 1]]])),
        dtype=torch.float64, device=device)
    for ia, a in enumerate(polesides):
        af = a + 2 * f
        xmax = torch.nn.functional.max_pool2d(
            ogm, kernel_size=[af, f], stride=1)
        xmax = torch.max(xmax[..., :-a-f], xmax[..., a+f:])

        ymax = torch.nn.functional.max_pool2d(
            ogm, kernel_size=[f, af], stride=1)
        ymax = torch.max(ymax[..., :-a-f, :], ymax[..., a+f:, :])
        
        kernel = torch.ones([1, 1, a, a], dtype=torch.float64, device=device) \
            / (a**2)

        score = (torch.nn.functional.conv2d(ogm[..., f:-f, f:-f], kernel) \
            - torch.max(xmax, ymax)).squeeze()

        accuscores[ia] = torch.nn.functional.max_pool2d(
            torch.nn.functional.pad(score, [a-1] * 4, 'constant', -1.0),
            kernel_size=a, stride=1) / 2.0 + 0.5
    accuscore = torch.max(accuscores, 0)[0]
    
    h = torch.zeros(
        tuple(polemapshape[:2]), dtype=torch.uint8, device=device)
    hmax = h.clone()
    z = h.clone()
    zmax = z.clone()
    for im, m in enumerate(accuscore):
        ispole = m >= minscore
        h += ispole
        ih = h > hmax
        hmax[ih] = h[ih]
        zmax[ih] = z[ih]
        h[ispole == False] = 0
        z[ispole == False] = im + 1

    accuscores = accuscores.cpu().numpy()
    accuscore = accuscore.cpu().numpy()
    hmax = hmax.cpu().numpy()
    zmax = zmax.cpu().numpy()

    ix, iy = np.where(hmax >= minheight / mapsize[2])
    poleness = np.zeros(polemapshape[:2])
    if ix.size > 0:
        for iix, iiy in zip(ix, iy):
            h = hmax[iix, iiy]
            z = zmax[iix, iiy]
            poleness[iix, iiy] = np.mean(accuscore[z:z+h, iix, iiy])

    peaks = skimage.feature.peak_local_max(
        poleness, min_distance=f, exclude_border=False, indices=False)
    label = skimage.measure.label(peaks, neighbors=8, background=False)
    regions = skimage.measure.regionprops(label, coordinates='rc')
    centroids = np.array([r.centroid for r in regions]) + 0.5

    normdist = scipy.stats.norm(0.0, normstd / mapsize[0])
    cellcoords = np.meshgrid(
        range(polemapshape[0]), range(polemapshape[1]), indexing='ij')
    cellcoords = np.stack(cellcoords, axis=2).astype(np.float) + 0.5
    
    poleparams = np.empty([centroids.shape[0], 6])
    optcentroids = centroids.copy()
    for ic in range(optcentroids.shape[0]):
        lastcentroid = np.full(2, np.inf)
        while np.linalg.norm(lastcentroid - optcentroids[ic]) \
                > dstop / mapsize[0]:
            lastcentroid = optcentroids[ic]
            d = np.linalg.norm(cellcoords - optcentroids[ic], axis=2)
            weights = np.tile(np.expand_dims(poleness * normdist.pdf(d), 2),
                [1, 1, 2])
            optcentroids[ic] = np.average(
                cellcoords, weights=weights, axis=(0, 1))

        ix, iy = np.floor(optcentroids[ic]).astype(np.int)
        if hmax[ix, iy] < minheight / mapsize[2]:
            optcentroids[ic] = centroids[ic]
            ix, iy = np.floor(centroids[ic]).astype(np.int)
        
        h = hmax[ix, iy]
        z = zmax[ix, iy]
        zstart = 0.0
        if z > 0:
            zstart = mapsize[2] * (np.interp(minscore, 
                accuscore[z-1:z+1, ix, iy], [z-1, z]) + 0.5)
        zend = occupancymap.shape[2] * mapsize[2]
        if z + h < polemapshape[2]:
            zend = mapsize[2] * (np.interp(minscore, 
                accuscore[z+h-1:z+h+1, ix, iy], [z+h-1, z+h]) + 0.5)
        
        sideweights = np.mean(accuscores[:, z:z+h, ix, iy], axis=1)
        sidelength = np.average(polesides, weights=sideweights) * mapsize[0]
        score = np.mean(np.average(
            accuscores[:, z:z+h, ix, iy], weights=sideweights, axis=0))
        
        x, y = mapsize[:2] * (optcentroids[ic] + f)
        poleparams[ic] = [x, y, zstart, zend, sidelength, score]

    poleparams = poleparams[np.flip(np.argsort(poleparams[:, -1]), axis=0), :]
    ip = 0
    while ip < poleparams.shape[0]:
        d = np.linalg.norm(poleparams[ip, :2] - poleparams[ip+1:, :2], axis=1) \
            - 0.5 * (poleparams[ip, 4] + poleparams[ip+1:, 4])
        poleparams = np.delete(
            poleparams, np.where(d < freelength)[0] + ip + 1, axis=0)
        ip += 1

    return poleparams
    