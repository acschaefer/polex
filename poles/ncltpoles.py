#!/usr/bin/env python

import copy
import datetime
import multiprocessing
import os
import shutil

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3
import progressbar
import scipy.interpolate
import scipy.special

import cluster
import mapping
import ndshow
import particlefilter
import poles
import pynclt
import util


plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'lmodern',
          'text.latex.unicode': True}

mapextent = np.array([30.0, 30.0, 5.0])
mapsize = np.full(3, 0.2)
mapshape = np.array(mapextent / mapsize, dtype=np.int)
mapinterval = 1.5
mapdistance = 1.5
remapdistance = 10.0
n_mapdetections = 3
n_locdetections = 2
n_localmaps = 3

poles.minscore = 0.6
poles.minheight = 1.0
poles.freelength = 0.5
poles.polesides = range(1, 7+1)

T_mc_r = pynclt.T_w_o
T_r_mc = util.invert_ht(T_mc_r)
T_m_mc = np.identity(4)
T_m_mc[:3, 3] = np.hstack([0.5 * mapextent[:2], 0.5])
T_mc_m = util.invert_ht(T_m_mc)
T_m_r = T_m_mc.dot(T_mc_r)
T_r_m = util.invert_ht(T_m_r)


def get_globalmapname():
    return 'globalmap_{:.0f}_{:.0f}_{:.0f}'.format(
        n_mapdetections, 10 * poles.minscore, poles.polesides[-1])


def get_locfileprefix():
    return 'localization_{:.0f}_{:.0f}_{:.0f}'.format(
        n_mapdetections, 10 * poles.minscore, poles.polesides[-1])


def get_localmapfile():
    return 'localmaps_{:.0f}_{:.0f}.npz'.format(
        10 * poles.minscore, poles.polesides[-1])


def get_evalfile():
    return 'evaluation_{:.0f}_{:.0f}.npz'.format(
        10 * poles.minscore, poles.polesides[-1])


def get_map_indices(session):
    distance = np.hstack([0.0, np.cumsum(np.linalg.norm(
        np.diff(session.T_w_r_gt_velo[:, :3, 3], axis=0), axis=1))])
    istart = []
    imid = []
    iend = []
    i = 0
    j = 0
    k = 0
    for id, d in enumerate(distance):
        if d >= i * mapinterval:
            istart.append(id)
            i += 1
        if d >= j * mapinterval + 0.5 * mapdistance:
            imid.append(id)
            j += 1
        if d > k * mapinterval + mapdistance:
            iend.append(id)
            k += 1
    return istart[:len(iend)], imid[:len(iend)], iend


def save_global_map():
    globalmappos = np.empty([0, 2])
    mapfactors = np.full(len(pynclt.sessions), np.nan)
    poleparams = np.empty([0, 6])
    for isession, s in enumerate(pynclt.sessions):
        print(s)
        session = pynclt.session(s)
        istart, imid, iend = get_map_indices(session)
        localmappos = session.T_w_r_gt_velo[imid, :2, 3]
        if globalmappos.size == 0:
            imaps = range(localmappos.shape[0])
        else:
            imaps = []
            for imap in range(localmappos.shape[0]):
                distance = np.linalg.norm(
                    localmappos[imap] - globalmappos, axis=1).min()
                if distance > remapdistance:
                    imaps.append(imap)
        globalmappos = np.vstack([globalmappos, localmappos[imaps]])
        mapfactors[isession] = np.true_divide(len(imaps), len(imid))

        with progressbar.ProgressBar(max_value=len(imaps)) as bar:
            for iimap, imap in enumerate(imaps):
                scans = []
                for iscan in range(istart[imap], iend[imap]):
                    xyz, _ = session.get_velo(iscan)
                    scan = o3.PointCloud()
                    scan.points = o3.Vector3dVector(xyz)
                    scans.append(scan)
                
                T_w_mc = np.identity(4)
                T_w_mc[:3, 3] = session.T_w_r_gt_velo[imid[imap], :3, 3]
                T_w_m = T_w_mc.dot(T_mc_m)
                T_m_w = util.invert_ht(T_w_m)
                T_m_r = np.matmul(
                    T_m_w, session.T_w_r_gt_velo[istart[imap]:iend[imap]])
                occupancymap = mapping.occupancymap(
                    scans, T_m_r, mapshape, mapsize)
                localpoleparams = poles.detect_poles(occupancymap, mapsize)
                localpoleparams[:, :2] += T_w_m[:2, 3]
                poleparams = np.vstack([poleparams, localpoleparams])
                bar.update(iimap)

    xy = poleparams[:, :2]
    a = poleparams[:, [4]]
    boxes = np.hstack([xy - 0.5 * a, xy + 0.5 * a])
    clustermeans = np.empty([0, 5])
    scores = []
    for ci in cluster.cluster_boxes(boxes):
        ci = list(ci)
        if len(ci) < n_mapdetections:
            continue
        clustermeans = np.vstack([clustermeans, np.average(
            poleparams[ci, :-1], axis=0, weights=poleparams[ci, -1])])
        scores.append(np.mean(poleparams[ci, -1]))
    clustermeans = np.hstack([clustermeans, np.array(scores).reshape([-1, 1])])
    globalmapfile = os.path.join('nclt', get_globalmapname() + '.npz')
    np.savez(globalmapfile, 
        polemeans=clustermeans, mapfactors=mapfactors, mappos=globalmappos)
    plot_global_map(globalmapfile)


def plot_global_map(globalmapfile):
    data = np.load(globalmapfile)
    x, y = data['polemeans'][:, :2].T
    plt.clf()
    plt.rcParams.update(params)
    plt.scatter(x, y, s=1, c='b', marker='.')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(globalmapfile[:-4] + '.svg')
    plt.savefig(globalmapfile[:-4] + '.pgf')
    print(data['mapfactors'])


def save_local_maps(sessionname, visualize=False):
    print(sessionname)
    session = pynclt.session(sessionname)
    util.makedirs(session.dir)
    istart, imid, iend = get_map_indices(session)
    maps = []
    with progressbar.ProgressBar(max_value=len(iend)) as bar:
        for i in range(len(iend)):
            scans = []
            for idx, val in enumerate(range(istart[i], iend[i])):
                xyz, _ = session.get_velo(val)
                scan = o3.PointCloud()
                scan.points = o3.Vector3dVector(xyz)
                scans.append(scan)

            T_w_mc = util.project_xy(
                session.T_w_r_odo_velo[imid[i]].dot(T_r_mc))
            T_w_m = T_w_mc.dot(T_mc_m)
            T_m_w = util.invert_ht(T_w_m)
            T_w_r = session.T_w_r_odo_velo[istart[i]:iend[i]]
            T_m_r = np.matmul(T_m_w, T_w_r)

            occupancymap = mapping.occupancymap(scans, T_m_r, mapshape, mapsize)
            poleparams = poles.detect_poles(occupancymap, mapsize)

            if visualize:
                cloud = o3.PointCloud()
                for T, scan in zip(T_w_r, scans):
                    s = copy.copy(scan)
                    s.transform(T)
                    cloud.points.extend(s.points)
                mapboundsvis = util.create_wire_box(mapextent, [0.0, 0.0, 1.0])
                mapboundsvis.transform(T_w_m)
                polevis = []
                for j in range(poleparams.shape[0]):
                    x, y, zs, ze, a = poleparams[j, :5]
                    pole = util.create_wire_box(
                        [a, a, ze - zs], color=[1.0, 1.0, 0.0])
                    T_m_p = np.identity(4)
                    T_m_p[:3, 3] = [x - 0.5 * a, y - 0.5 * a, zs]
                    pole.transform(T_w_m.dot(T_m_p))
                    polevis.append(pole)
                o3.draw_geometries(polevis + [cloud, mapboundsvis])

            map = {'poleparams': poleparams, 'T_w_m': T_w_m,
                'istart': istart[i], 'imid': imid[i], 'iend': iend[i]}
            maps.append(map)
            bar.update(i)
    np.savez(os.path.join(session.dir, get_localmapfile()), maps=maps)


def view_local_maps(sessionname):
    sessiondir = os.path.join('nclt', sessionname)
    session = pynclt.session(sessionname)
    maps = np.load(os.path.join(sessiondir, get_localmapfile()))['maps']
    for i, map in enumerate(maps):
        print('Map #{}'.format(i))
        mapboundsvis = util.create_wire_box(mapextent, [0.0, 0.0, 1.0])
        mapboundsvis.transform(map['T_w_m'])
        polevis = []
        for poleparams in map['poleparams']:
            x, y, zs, ze, a = poleparams[:5]
            pole = util.create_wire_box(
                [a, a, ze - zs], color=[1.0, 1.0, 0.0])
            T_m_p = np.identity(4)
            T_m_p[:3, 3] = [x - 0.5 * a, y - 0.5 * a, zs]
            pole.transform(map['T_w_m'].dot(T_m_p))
            polevis.append(pole)

        accucloud = o3.PointCloud()
        for j in range(map['istart'], map['iend']):
            points, intensities = session.get_velo(j)
            cloud = o3.PointCloud()
            cloud.points = o3.Vector3dVector(points)
            cloud.colors = o3.Vector3dVector(
                util.intensity2color(intensities / 255.0))
            cloud.transform(session.T_w_r_odo_velo[j])
            accucloud.points.extend(cloud.points)
            accucloud.colors.extend(cloud.colors)

        o3.draw_geometries([accucloud, mapboundsvis] + polevis)


def evaluate_matches():
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans'][:, :2]
    kdtree = scipy.spatial.cKDTree(polemap[:, :2], leafsize=10)
    maxdist = 0.5
    n_matches = np.zeros(len(pynclt.sessions))
    n_all = np.zeros(len(pynclt.sessions))
    for i, sessionname in enumerate(pynclt.sessions):
        sessiondir = os.path.join('nclt', sessionname)
        session = pynclt.session(sessionname)
        maps = np.load(os.path.join(sessiondir, get_localmapfile()))['maps']
        for map in maps:
            n = map['poleparams'].shape[0]
            n_all[i] += n
            polepos_m = np.hstack(
                [map['poleparams'][:, :2], np.zeros([n, 1]), np.ones([n, 1])]).T
            T_w_m = session.T_w_r_gt_velo[map['imid']].dot(T_r_m)
            polepos_w = T_w_m.dot(polepos_m)
            
            plt.scatter(polemap[:, 0], polemap[:, 1], color='k')
            plt.scatter(polepos_w[0,:], polepos_w[1,:], color='r')
            plt.show()
            
            dist, _ = kdtree.query(
                polepos_w[:2].T, k=1, distance_upper_bound=maxdist)
            n_matches[i] += np.sum(np.isfinite(dist))
        print('{}: {}'.format(
            sessionname, np.true_divide(n_matches[i], n_all[i])))


def localize(sessionname, visualize=False):
    print(sessionname)
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans'][:, :2]
    polevar = 1.50
    session = pynclt.session(sessionname)
    locdata = np.load(os.path.join(session.dir, get_localmapfile()))['maps']
    polepos_m = []
    polepos_w = []
    for i in range(len(locdata)):
        n = locdata[i]['poleparams'].shape[0]
        pad = np.hstack([np.zeros([n, 1]), np.ones([n, 1])])
        polepos_m.append(np.hstack([locdata[i]['poleparams'][:, :2], pad]).T)
        polepos_w.append(locdata[i]['T_w_m'].dot(polepos_m[i]))
    istart = 0
    # igps = np.searchsorted(session.t_gps, session.t_relodo[istart]) + [-4, 1]
    # igps = np.clip(igps, 0, session.gps.shape[0] - 1)
    # T_w_r_start = pynclt.T_w_o
    # T_w_r_start[:2, 3] = np.mean(session.gps[igps], axis=0)
    T_w_r_start = util.project_xy(
        session.get_T_w_r_gt(session.t_relodo[istart]).dot(T_r_mc)).dot(T_mc_r)
    filter = particlefilter.particlefilter(5000, 
        T_w_r_start, 2.5, np.radians(5.0), polemap, polevar, T_w_o=T_mc_r)
    filter.estimatetype = 'best'
    filter.minneff = 0.5

    if visualize:
        plt.ion()
        figure = plt.figure()
        nplots = 1
        mapaxes = figure.add_subplot(nplots, 1, 1)
        mapaxes.set_aspect('equal')
        mapaxes.scatter(polemap[:, 0], polemap[:, 1], s=5, c='b', marker='s')
        x_gt, y_gt = session.T_w_r_gt[::20, :2, 3].T
        mapaxes.plot(x_gt, y_gt, 'g')
        particles = mapaxes.scatter([], [], s=1, c='r')
        arrow = mapaxes.arrow(0.0, 0.0, 1.0, 0.0, length_includes_head=True, 
            head_width=0.7, head_length=1.0, color='k')
        arrowdata = np.hstack(
            [arrow.get_xy(), np.zeros([8, 1]), np.ones([8, 1])]).T
        locpoles = mapaxes.scatter([], [], s=30, c='k', marker='x')
        viewoffset = 25.0

        # weightaxes = figure.add_subplot(nplots, 1, 2)
        # gridsize = 50
        # offset = 5.0
        # visfilter = particlefilter.particlefilter(gridsize**2, 
        #     np.identity(4), 0.0, 0.0, polemap, 
        #     polevar, T_w_o=pynclt.T_w_o)
        # gridcoord = np.linspace(-offset, offset, gridsize)
        # x, y = np.meshgrid(gridcoord, gridcoord)
        # dxy = np.hstack([x.reshape([-1, 1]), y.reshape([-1, 1])])
        # weightimage = weightaxes.matshow(np.zeros([gridsize, gridsize]), 
        #     extent=(-offset, offset, -offset, offset))
        # histaxes = figure.add_subplot(nplots, 1, 3)

    imap = 0
    while imap < locdata.shape[0] - 1 and \
            session.t_velo[locdata[imap]['iend']] < session.t_relodo[istart]:
        imap += 1
    T_w_r_est = np.full([session.t_relodo.size, 4, 4], np.nan)
    with progressbar.ProgressBar(max_value=session.t_relodo.size) as bar:
        for i in range(istart, session.t_relodo.size):
            relodocov = np.empty([3, 3])
            relodocov[:2, :2] = session.relodocov[i, :2, :2]
            relodocov[:, 2] = session.relodocov[i, [0, 1, 5], 5]
            relodocov[2, :] = session.relodocov[i, 5, [0, 1, 5]]
            filter.update_motion(session.relodo[i], relodocov * 2.0**2)
            T_w_r_est[i] = filter.estimate_pose()
            t_now = session.t_relodo[i]
            if imap < locdata.shape[0]:
                t_end = session.t_velo[locdata[imap]['iend']]
                if t_now >= t_end:
                    imaps = range(imap, np.clip(imap-n_localmaps, -1, None), -1)
                    xy = np.hstack([polepos_w[j][:2] for j in imaps]).T
                    a = np.vstack([ld['poleparams'][:, [4]] \
                        for ld in locdata[imaps]])
                    boxes = np.hstack([xy - 0.5 * a, xy + 0.5 * a])
                    ipoles = set(range(polepos_w[imap].shape[1]))
                    iactive = set()
                    for ci in cluster.cluster_boxes(boxes):
                        if len(ci) >= n_locdetections:
                            iactive |= set(ipoles) & ci
                    iactive = list(iactive)
                    # print('{}.'.format(
                    #     len(iactive) - polepos_w[imap].shape[1]))
                    if iactive:
                        t_mid = session.t_velo[locdata[imap]['imid']]
                        T_w_r_mid = util.project_xy(session.get_T_w_r_odo(
                            t_mid).dot(T_r_mc)).dot(T_mc_r)
                        T_w_r_now = util.project_xy(session.get_T_w_r_odo(
                            t_now).dot(T_r_mc)).dot(T_mc_r)
                        T_r_now_r_mid = util.invert_ht(T_w_r_now).dot(T_w_r_mid)
                        polepos_r_now = T_r_now_r_mid.dot(T_r_m).dot(
                            polepos_m[imap][:, iactive])
                        filter.update_measurement(polepos_r_now[:2].T)
                        T_w_r_est[i] = filter.estimate_pose()
                        if visualize:
                            polepos_w_est = T_w_r_est[i].dot(polepos_r_now)
                            locpoles.set_offsets(polepos_w_est[:2].T)

                            # T_w_r_gt_now = session.get_T_w_r_gt(t_now)
                            # T_w_r_gt_now = np.tile(
                            #     T_w_r_gt_now, [gridsize**2, 1, 1])
                            # T_w_r_gt_now[:, :2, 3] += dxy
                            # visfilter.particles = T_w_r_gt_now
                            # visfilter.weights[:] = 1.0 / visfilter.count
                            # visfilter.update_measurement(
                            #     polepos_r_now[:2].T, resample=False)
                            # weightimage.set_array(np.flipud(
                            #     visfilter.weights.reshape(
                            #         [gridsize, gridsize])))
                            # weightimage.autoscale()
                    imap += 1
            
            if visualize:
                particles.set_offsets(filter.particles[:, :2, 3])
                arrow.set_xy(T_w_r_est[i].dot(arrowdata)[:2].T)
                x, y = T_w_r_est[i, :2, 3]
                mapaxes.set_xlim(left=x - viewoffset, right=x + viewoffset)
                mapaxes.set_ylim(bottom=y - viewoffset, top=y + viewoffset)
                # histaxes.cla()
                # histaxes.hist(filter.weights, 
                #     bins=50, range=(0.0, np.max(filter.weights)))
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
            bar.update(i)
    filename = os.path.join(session.dir, get_locfileprefix() \
        + datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.npz'))
    np.savez(filename, T_w_r_est=T_w_r_est)


def plot_trajectories():
    trajectorydir = os.path.join(
        pynclt.resultdir, 'trajectories_est_{:.0f}_{:.0f}_{:.0f}'.format(
            n_mapdetections, 10 * poles.minscore, poles.polesides[-1]))
    pgfdir = os.path.join(trajectorydir, 'pgf')
    util.makedirs(trajectorydir)
    util.makedirs(pgfdir)
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans']
    plt.rcParams.update(params)
    for sessionname in pynclt.sessions:
        try:
            session = pynclt.session(sessionname)
            files = [file for file \
                in os.listdir(os.path.join(pynclt.resultdir, sessionname)) \
                    if file.startswith('localization_3_6_7_2019-07')]
                    # if file.startswith(get_locfileprefix())]
            for file in files:
                T_w_r_est = np.load(os.path.join(
                    pynclt.resultdir, sessionname, file))['T_w_r_est']
                plt.clf()
                plt.scatter(polemap[:, 0], polemap[:, 1], 
                    s=1, c='b', marker='.')
                plt.plot(session.T_w_r_gt[::20, 0, 3], 
                    session.T_w_r_gt[::20, 1, 3], color=(0.5, 0.5, 0.5))
                plt.plot(T_w_r_est[::20, 0, 3], T_w_r_est[::20, 1, 3], 'r')
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.gcf().subplots_adjust(
                    bottom=0.13, top=0.98, left=0.145, right=0.98)
                filename = sessionname + file[18:-4]
                plt.savefig(os.path.join(trajectorydir, filename + '.svg'))
                plt.savefig(os.path.join(pgfdir, filename + '.pgf'))
        except:
            pass


def evaluate():
    stats = []
    for sessionname in pynclt.sessions:
        files = [file for file \
            in os.listdir(os.path.join(pynclt.resultdir, sessionname)) \
                if file.startswith('localization_3_6_7_2019-07')]
                # if file.startswith(get_locfileprefix())]
        files.sort()
        session = pynclt.session(sessionname)
        cumdist = np.hstack([0.0, np.cumsum(np.linalg.norm(np.diff(
            session.T_w_r_gt[:, :3, 3], axis=0), axis=1))])
        t_eval = scipy.interpolate.interp1d(
            cumdist, session.t_gt)(np.arange(0.0, cumdist[-1], 1.0))
        T_w_r_gt = np.stack([util.project_xy(
                session.get_T_w_r_gt(t).dot(T_r_mc)).dot(T_mc_r) \
                    for t in t_eval])
        T_gt_est = []
        for file in files:
            T_w_r_est = np.load(os.path.join(
                pynclt.resultdir, sessionname, file))['T_w_r_est']
            T_w_r_est_interp = np.empty([len(t_eval), 4, 4])
            iodo = 1
            for ieval in range(len(t_eval)):
                while session.t_relodo[iodo] < t_eval[ieval]:
                    iodo += 1
                T_w_r_est_interp[ieval] = util.interpolate_ht(
                    T_w_r_est[iodo-1:iodo+1], 
                    session.t_relodo[iodo-1:iodo+1], t_eval[ieval])
            T_gt_est.append(
                np.matmul(util.invert_ht(T_w_r_gt), T_w_r_est_interp))
        T_gt_est = np.stack(T_gt_est)
        lonerror = np.mean(np.mean(np.abs(T_gt_est[..., 0, 3]), axis=-1))
        laterror = np.mean(np.mean(np.abs(T_gt_est[..., 1, 3]), axis=-1))
        poserrors = np.linalg.norm(T_gt_est[..., :2, 3], axis=-1)
        poserror = np.mean(np.mean(poserrors, axis=-1))
        posrmse = np.mean(np.sqrt(np.mean(poserrors**2, axis=-1)))
        angerrors = np.degrees(np.abs(
            np.array([util.ht2xyp(T)[:, 2] for T in T_gt_est])))
        angerror = np.mean(np.mean(angerrors, axis=-1))
        angrmse = np.mean(np.sqrt(np.mean(angerrors**2, axis=-1)))
        stats.append({'session': sessionname, 'lonerror': lonerror, 
            'laterror': laterror, 'poserror': poserror, 'posrmse': posrmse,
            'angerror': angerror, 'angrmse': angrmse, 'T_gt_est': T_gt_est})
    np.savez(os.path.join(pynclt.resultdir, get_evalfile()), stats=stats)
    
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    print('session \t f\te_pos \trmse_pos \te_ang \te_rmse')
    row = '{session} \t{f} \t{poserror} \t{posrmse} \t{angerror} \t{angrmse}'
    for i, stat in enumerate(stats):
        print(row.format(
            session=stat['session'],
            f=mapdata['mapfactors'][i] * 100.0,
            poserror=stat['poserror'],
            posrmse=stat['posrmse'],
            angerror=stat['angerror'],
            angrmse=stat['angrmse']))


if __name__ == '__main__':
    poles.minscore = 0.6
    poles.polesides = range(1, 7+1)
    save_global_map()
    session = '2012-01-08'
    save_local_maps(session)
    localize(session)
    plot_trajectories()
    evaluate()    
