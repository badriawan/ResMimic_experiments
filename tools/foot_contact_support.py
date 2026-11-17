"""Priority 1: separate, stance-aware URDF reference candidate, without dynamics.

Run with existing gmr_omomo Python, -I -B. Never overwrites Stage 11 files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from foot_runtime import install, REPO, WORK
install()
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(REPO / 'legged_gym/legged_gym/scripts'))
from convert_gmr_qpos import forward_kinematics, foot_clearance, clearance_summary, origin, FEET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x)
    if not x.size:
        return None
    return dict(min=float(x.min()), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(x.max()))


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def phase(marker, fps):
    """Conservative inferred stance; unclassified frames are explicitly unknown."""
    smooth = gaussian_filter1d(marker, 1.0, axis=0, mode='nearest')
    speed = np.linalg.norm(np.gradient(smooth, 1 / fps, axis=0), axis=1)
    floor = float(np.percentile(smooth[:, 2], 10))
    height = smooth[:, 2] - floor
    mask = np.zeros(len(marker), bool)
    active = False
    for i in range(len(marker)):
        if active:
            active = bool(height[i] < .025 and speed[i] < .25)
        else:
            active = bool(height[i] < .015 and speed[i] < .12)
        mask[i] = active
    for start, stop in runs(mask):
        if stop - start < 3:
            mask[start:stop] = False
    return mask, speed, height, floor


class Chain:
    """Cached URDF chain; evaluate only the joints relevant to a foot or hand."""
    def __init__(self, urdf, names, tip):
        self.indices = []
        by_child = {j.find('child').get('link'): j for j in urdf.findall('joint')}
        joints = []
        while tip != 'pelvis':
            j = by_child[tip]
            joints.append(j)
            tip = j.find('parent').get('link')
        self.steps = []
        for j in joints[::-1]:
            xyz, rot = origin(j)
            idx = names.index(j.get('name')) if j.get('type') != 'fixed' else None
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if idx is not None else None
            if idx is not None:
                self.indices.append(idx)
            self.steps.append((xyz, rot, idx, axis))

    def fk(self, q, rp, rr):
        p, r = rp.copy(), rr.copy()
        for xyz, rot, idx, axis in self.steps:
            p = p + r @ xyz
            r = r @ rot
            if idx is not None:
                r = r @ Rotation.from_rotvec(q[idx] * axis).as_matrix()
        return p, r


def offset_curve(n, mask, values, baseline):
    """Constant offsets in stance and C1 smoothstep transitions in the gaps."""
    segments = runs(mask)
    correction = np.zeros_like(baseline)
    anchors = []
    for start, stop in segments:
        target = np.median(values[start:stop], axis=0)
        correction[start:stop] = target - baseline[start:stop]
        anchors.append(target)
    if not segments:
        raise ValueError('No credible stance segments; cannot infer side ground offset')
    correction[:segments[0][0]] = correction[segments[0][0]]
    correction[segments[-1][1]:] = correction[segments[-1][1] - 1]
    for (_, a), (b, _) in zip(segments[:-1], segments[1:]):
        u = np.arange(1, b - a + 1) / (b - a + 1)
        w = u * u * (3 - 2 * u)
        correction[a:b] = ((1 - w[:, None]) * correction[a - 1]
                           + w[:, None] * correction[b])
    return baseline + correction, anchors


def sole_offset(urdf, foot, rot):
    """Conservative downward extent, checking cylinder and capsule variants."""
    lows = []
    for collision in urdf.find("link[@name='%s']" % foot).findall('collision'):
        xyz, r = origin(collision)
        geom = list(collision.find('geometry'))[0]
        if geom.tag != 'cylinder':
            raise ValueError('This pinned foot candidate expects two URDF cylinders')
        zaxis = abs((rot @ r)[2, 2])
        radius, half = float(geom.get('radius')), float(geom.get('length')) / 2
        center = (rot @ xyz)[2]
        lows.extend([center - half * zaxis - radius,
                     center - half * zaxis - radius * np.sqrt(max(0, 1 - zaxis*zaxis))])
    return min(lows)


def solve_chain(chain, baseline, previous, rp, rr, target_p, target_r, lower, upper, indices):
    def residual(values):
        q = baseline.copy()
        q[indices] = values
        p, r = chain.fk(q, rp, rr)
        return np.r_[(p - target_p) * 1000,
                     Rotation.from_matrix(target_r.T @ r).as_rotvec() * 100,
                     (values - baseline[indices]) * .01,
                     (values - previous) * .01]
    result = least_squares(residual, np.clip(previous, lower[indices]+1e-9, upper[indices]-1e-9),
                           bounds=(lower[indices], upper[indices]),
                           max_nfev=400, ftol=1e-8, xtol=1e-8, gtol=1e-7)
    return result.x, result.success, result.nfev


def evaluate(m, urdf, names, masks, target_xy=None):
    fps = m['fps']
    lp, lr = forward_kinematics(urdf, m['dof_pos'], names, m['link_body_list'])
    rr = Rotation.from_quat(m['root_rot']).as_matrix()
    world = m['root_pos'][:, None] + np.einsum('tij,tbj->tbi', rr, lp)
    clear = foot_clearance(urdf, lp, lr, m['link_body_list'], m['root_pos'], rr)
    feet = {}
    for side, foot in enumerate(FEET):
        xy = world[:, m['link_body_list'].index(foot), :2]
        mask = masks[:, side]
        details = []
        speeds = []
        for start, stop in runs(mask):
            displacement = np.linalg.norm(xy[start:stop] - xy[start], axis=1)
            speed = np.linalg.norm(np.diff(xy[start:stop], axis=0), axis=1) * fps
            speeds.extend(speed)
            details.append(dict(start_frame=int(start), stop_frame_exclusive=int(stop),
                                max_drift_from_first_m=float(displacement.max()),
                                path_length_m=float(speed.sum()/fps)))
        feet[foot] = dict(stance_frames=int(mask.sum()), segments=details,
                          stance_speed_m_s=stats(speeds),
                          stance_gap_m={key: stats(value[mask]) for key, value in clear[foot].items()},
                          all_frame_gap_m={key: stats(value) for key, value in clear[foot].items()})
    return dict(feet=feet, clearance=clearance_summary(clear)), world, lr, clear

