"""Independent reference validation math; no candidate-generation imports."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from converter_common import REPO
sys.path.insert(0,str(REPO/'legged_gym/legged_gym/scripts'))
from convert_gmr_qpos import forward_kinematics, foot_clearance, origin, FEET

def stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    return dict(min=float(values.min()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))

def linear(values, times, fps):
    coordinate = np.clip(np.asarray(times) * fps, 0, len(values) - 1)
    lo = np.minimum(coordinate.astype(int), len(values) - 2)
    w = coordinate - lo
    shape = (len(w),) + (1,) * (values.ndim - 1)
    return (1 - w.reshape(shape)) * values[lo] + w.reshape(shape) * values[lo + 1]

def pose_samples(motion, times):
    return dict(root_pos=linear(motion['root_pos'], times, motion['fps']),
                root_rot=Slerp(np.arange(len(motion['root_rot'])) / motion['fps'],
                               Rotation.from_quat(motion['root_rot']))(times).as_quat(),
                dof_pos=linear(motion['dof_pos'], times, motion['fps']))

def world_fk(motion, urdf, names, bodies):
    lp, lr = forward_kinematics(urdf, motion['dof_pos'], names, bodies)
    rr = Rotation.from_quat(motion['root_rot']).as_matrix()
    world = motion['root_pos'][:, None] + np.einsum('tij,tbj->tbi', rr, lp)
    return lp, lr, rr, world

def rotation_delta(a, b):
    ar = Rotation.from_quat(a).as_matrix()
    br = Rotation.from_quat(b).as_matrix()
    return Rotation.from_matrix(np.swapaxes(ar, 1, 2) @ br).magnitude()

def contact_assessment(sampled, urdf, names, bodies, times):
    lp, lr, rr, world = world_fk(sampled, urdf, names, bodies)
    clearance = foot_clearance(urdf, lp, lr, bodies, sampled['root_pos'], rr)
    dt = np.diff(times)
    support = np.zeros((len(times), 2), dtype=bool)
    feet = {}
    all_probe_max = 0.
    all_probe_p95 = 0.
    all_ankle_max = 0.
    all_ankle_p95 = 0.
    ankle_traces = []
    probe_traces = []
    near_traces = []
    for side, foot in enumerate(FEET):
        variants = clearance[foot]
        gap = np.minimum(variants['cylinder'], variants['capsule_surrogate'])
        near = (gap[:-1] < .005) | (gap[1:] < .005)
        xyz = world[:, bodies.index(foot)]
        speed = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) / dt
        ankle_stats = stats(speed[near])
        support[:, side] = np.maximum(abs(variants['cylinder']),
                                     abs(variants['capsule_surrogate'])) <= .005
        probe_speeds = []
        per_probe = []
        probe_interval_max = np.zeros(len(times) - 1)
        wr = rr @ lr[foot]
        for collision in urdf.find("link[@name='%s']" % foot).findall('collision'):
            cp, cr = origin(collision)
            geom = list(collision.find('geometry'))[0]
            if geom.tag != 'cylinder':
                raise ValueError('Pinned G1 foot model must contain cylinders')
            radius = float(geom.get('radius'))
            half = float(geom.get('length')) / 2
            for sign in [-1, 0, 1]:
                point = xyz + np.einsum('tij,j->ti', wr, cp + sign * half * cr[:, 2])
                pgap = point[:, 2] - radius
                p_near = (pgap[:-1] < .005) | (pgap[1:] < .005)
                pv = np.linalg.norm(np.diff(point[:, :2], axis=0), axis=1) / dt
                probe_speeds.extend(pv[p_near])
                probe_interval_max = np.maximum(probe_interval_max, np.where(p_near, pv, 0.))
                per_probe.append(dict(collision=collision.get('name'), axis_sign=sign,
                                      near_floor_intervals=int(p_near.sum()),
                                      speed_m_s=stats(pv[p_near])))
        probe_stats = stats(probe_speeds)
        if ankle_stats is not None:
            all_ankle_max = max(all_ankle_max, ankle_stats['max'])
            all_ankle_p95 = max(all_ankle_p95, ankle_stats['p95'])
        if probe_stats is not None:
            all_probe_max = max(all_probe_max, probe_stats['max'])
            all_probe_p95 = max(all_probe_p95, probe_stats['p95'])
        feet[foot] = dict(gap_m={k: stats(v) for k, v in variants.items()},
                          near_floor_intervals=int(near.sum()), near_floor_ankle_speed_m_s=ankle_stats,
                          near_floor_ankle_intervals_above_0_1m_s=int((near & (speed > .1)).sum()),
                          near_floor_probe_speed_m_s=probe_stats,
                          near_floor_probe_intervals_above_0_1m_s=int((np.asarray(probe_speeds) > .1).sum()),
                          probes=per_probe)
        ankle_traces.append(speed)
        probe_traces.append(probe_interval_max)
        near_traces.append(near)
    unsupported = ~support.any(axis=1)
    minimum = min(v.min() for variant in clearance.values() for v in variant.values())
    gates = dict(penetration_2mm=bool(minimum >= -.002), support_every_sample=not bool(unsupported.any()),
                 near_floor_ankle_max_0_1m_s=all_ankle_max <= .1,
                 near_floor_ankle_p95_0_05m_s=all_ankle_p95 <= .05,
                 near_floor_probe_max_0_1m_s=all_probe_max <= .1,
                 near_floor_probe_p95_0_05m_s=all_probe_p95 <= .05)
    result = dict(samples=len(times), nominal_sample_rate_hz=float(1 / np.median(dt)),
                  minimum_gap_m=float(minimum), unsupported_samples=int(unsupported.sum()),
                  unsupported_times_s=np.asarray(times)[unsupported].tolist(), feet=feet,
                  definition='All intervals: either endpoint native-cylinder or capsule gap <5mm; no phase mask. Capsule probes are fixed centres and both axis endpoints, with sphere-radius vertical gap.',
                  endpoint_policy='Actual loader below endpoint; exact last exported pose appended, avoiding API wrap',
                  limitation='Finite time sampling; not continuous-time proof or measured physical slip')
    traces = dict(times_s=np.asarray(times), feet_world_pos=world[:, [bodies.index(f) for f in FEET]],
                  cylinder_gap=np.stack([clearance[f]['cylinder'] for f in FEET], axis=1),
                  capsule_gap=np.stack([clearance[f]['capsule_surrogate'] for f in FEET], axis=1),
                  ankle_xy_speed=np.stack(ankle_traces, axis=1),
                  capsule_probe_near_floor_max_xy_speed=np.stack(probe_traces, axis=1),
                  ankle_near_floor_interval=np.stack(near_traces, axis=1))
    return gates, result, (lp, lr, rr, world), traces
