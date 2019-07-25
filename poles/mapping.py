#!/usr/bin/env python

import copy
import os
import psutil
import shutil
import subprocess

import numpy as np
import open3d as o3
import scipy.special

import raytracing as rt


tmpdir = 'tmp'
scanformat = 'ply'
occupancythreshold = 0.05


# Scan points are specified with respect to the sensor coordinate frame.
# Poses are specified with respect to the map coordinate frame.
def occupancymap(scans, poses, mapshape, mapsize):
    map = rt.gridmap(mapshape, mapsize)
    for scan, pose in zip(scans, poses):
        scan_map = copy.copy(scan)
        scan_map.transform(pose)
        points = np.asarray(scan_map.points)
        rt.trace3d(np.tile(pose[:3, 3], [points.shape[0], 1]), points, map)

    reflectionmap = map.reflectionmap()
    reflectivity = reflectionmap[np.isfinite(reflectionmap)]
    if reflectivity.size == 0:
        occupancymap = np.zeros(map.shape)
    else:
        mean = np.mean(reflectivity)
        var = np.var(reflectivity)
        alpha = -mean * ((mean**2.0 - mean) / var + 1.0)
        beta = mean - 1.0 + (mean - 2.0 * mean**2.0 + mean**3.0) / var
        prior = 1.0 - scipy.special.betainc(alpha, beta, occupancythreshold)
        occupancymap = 1.0 - scipy.special.betainc(
            map.hits + alpha, map.misses + beta, occupancythreshold)
        
    return occupancymap


# Scan points are specified with respect to the sensor coordinate frame.
def export(scans, poses, dir):
    scannames = []
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    for i in range(len(scans)):
        scanname = 'scan{:03d}'.format(i)
        scannames.append(scanname)
        o3.write_point_cloud(
            os.path.join(dir, scanname + '.' + scanformat), scans[i])

        rlconversion = np.array([1, -1, 1])
        trans = poses[i, :3, 3] * rlconversion
        rot = np.degrees(t3.euler.mat2euler(poses[i, :3, :3])) * -rlconversion
        pose = np.vstack([trans, rot])
        np.savetxt(
            os.path.join(dir, scanname + '.pose'), pose, delimiter=' ')

    process = subprocess.Popen(['pose2frames', '.'], cwd=dir)
    if process.wait() != 0:
        raise Exception('Module \"pose2frames\" failed.')


# Scan points are specified with respect to the sensor coordinate frame.
def detect_dynamic_points(scans, poses):
    export(scans, poses, tmpdir)

    maskdir = 'mask'
    process = subprocess.Popen(['peopleremover', 
        '-f', scanformat,
        '--voxel-size', '0.02',
        '--fuzz', '0.1',
        '--min-cluster-size', '5',
        '--maxrange-method', 'normals',
        '--maskdir', maskdir,
        '-j', '{}'.format(psutil.cpu_count()),
        '.'], cwd=tmpdir)
    if process.wait() != 0:
        raise Exception('Module \"peopleremover\" failed.')

    dynamic = []
    for i in range(len(scans)):
        scanname = 'scan{:03d}'.format(i)
        maskfilename = os.path.join(tmpdir, maskdir, scanname + '.mask')
        dynamic.append(np.loadtxt(maskfilename, dtype=np.bool))

    return dynamic
