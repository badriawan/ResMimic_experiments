"""Opt-in SLERP adapter for reference validation in the current process only.

No checkout files or global torch-utils bindings are modified. This copy is scoped to conversion validation. FPS-sensitive deltas, rewards,
and checkpoint compatibility remain outside this mathematical adapter.
"""
import torch


def reference_slerp(q0, q1, t):
    """Normalized shortest-path interpolation, including tiny-angle endpoints."""
    if q0.shape != q1.shape or q0.shape[-1] != 4 or t.shape != q0.shape[:-1]:
        raise ValueError('Expected matching (...,4) quaternion arrays and (...) blends')
    tiny = torch.finfo(q0.dtype).tiny
    a = q0 / torch.linalg.vector_norm(q0, dim=-1, keepdim=True).clamp_min(tiny)
    b = q1 / torch.linalg.vector_norm(q1, dim=-1, keepdim=True).clamp_min(tiny)
    dot = (a * b).sum(dim=-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(0, 1)
    w = t.unsqueeze(-1).clamp(0, 1)
    # The tiny-angle branch MUST retain the actual blend, not a fixed midpoint.
    lerp = (1 - w) * a + w * b
    safe_dot = dot.clamp(max=1 - torch.finfo(q0.dtype).eps)
    theta = torch.acos(safe_dot)
    denominator = torch.sin(theta).clamp_min(tiny)
    curved = (torch.sin((1 - w) * theta) * a + torch.sin(w * theta) * b) / denominator
    result = torch.where(dot > 1 - 1e-6, lerp, curved)
    result = result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(tiny)
    result = torch.where(w <= 0, a, result)
    result = torch.where(w >= 1, b, result)
    return result


def install_reference_interpolation():
    """Return a restore token; change only the two loader-module bindings."""
    from pose.utils import motion_lib_pkl, motion_lib_hoi
    token = [(motion_lib_pkl, motion_lib_pkl.slerp),
             (motion_lib_hoi, motion_lib_hoi.slerp)]
    motion_lib_pkl.slerp = reference_slerp
    motion_lib_hoi.slerp = reference_slerp
    return token


def restore_reference_interpolation(token):
    for module, original in token:
        module.slerp = original


def self_test():
    """Compare tiny, antipodal, regular, and roundoff cases against SciPy."""
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    fixtures = [
        ('identity', np.array([0., 0., 0., 1.]), np.array([0., 0., 0., 1.])),
        ('tiny_1e-8', Rotation.from_rotvec([.2, -.1, .3]).as_quat(),
         (Rotation.from_rotvec([.2, -.1, .3]) * Rotation.from_rotvec([1e-8, 0., 0.])).as_quat()),
        ('tiny_8e-4', Rotation.from_rotvec([.1, -.5, .7]).as_quat(),
         (Rotation.from_rotvec([.1, -.5, .7]) * Rotation.from_rotvec([0., 8e-4, 0.])).as_quat()),
        ('near_antipodal_sign', Rotation.from_rotvec([.2, -.3, .4]).as_quat(),
         -(Rotation.from_rotvec([.2, -.3, .4]) * Rotation.from_rotvec([1e-7, 0., 0.])).as_quat()),
        ('exact_opposite_sign', Rotation.from_rotvec([.2, -.3, .4]).as_quat(),
         -Rotation.from_rotvec([.2, -.3, .4]).as_quat()),
        ('half_turn', np.array([0., 0., 0., 1.]), Rotation.from_rotvec([np.pi, 0., 0.]).as_quat()),
        ('regular', Rotation.from_rotvec([.3, -.2, .1]).as_quat(), Rotation.from_rotvec([1.2, -.8, .4]).as_quat()),
        ('roundoff_nonunit', Rotation.from_rotvec([-.7, .2, .1]).as_quat() * (1 + 2e-7),
         Rotation.from_rotvec([-.5, .3, .2]).as_quat() * (1 - 2e-7)),
    ]
    times = np.array([0., .25, .5, .75, 1.])
    results = []
    for dtype in [torch.float32, torch.float64]:
        for name, q0, q1 in fixtures:
            a = torch.tensor(np.tile(q0, (len(times), 1)), dtype=dtype)
            b = torch.tensor(np.tile(q1, (len(times), 1)), dtype=dtype)
            blend = torch.tensor(times, dtype=dtype)
            result = reference_slerp(a, b, blend).numpy()
            expected = Slerp([0., 1.], Rotation.from_quat(np.stack([q0, q1])))(times)
            angles = (expected.inv() * Rotation.from_quat(result)).magnitude()
            norm_error = abs(np.linalg.norm(result, axis=1) - 1)
            tolerance = 2e-6 if dtype == torch.float32 else 2e-10
            passed = bool(np.isfinite(result).all() and angles.max() < tolerance
                          and norm_error.max() < (2e-7 if dtype == torch.float32 else 1e-14))
            results.append(dict(name=name, dtype=str(dtype), max_angle_error_rad=float(angles.max()),
                                max_norm_error=float(norm_error.max()), passed=passed))
    return dict(status='PASS' if all(v['passed'] for v in results) else 'FAIL',
                cases=results, blend_values=times.tolist(),
                scope='Quaternion interpolation math only; does not establish policy or FPS compatibility')


if __name__ == '__main__':
    import json
    report = self_test()
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)
