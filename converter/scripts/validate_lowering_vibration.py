"""Independent validation of a lowering-phase reference edit.

All collision poses come from the actual checkout loader trace.  This tests
reference geometry only; it does not step physics or run a policy.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from converter_common import ROOT, VIBRATION_DATA, install, sha, write_json
from lift_hold_validation_common import (
    contact_assessment, linear, pose_samples, rotation_delta, stats, world_fk,
)
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

HANDS = ('left_rubber_hand', 'right_rubber_hand')
POSE_KEYS = ('root_pos', 'root_rot', 'dof_pos', 'local_body_pos')


def lowering_interval(obj, fps):
    """Infer downward object motion from the unchanged exported trajectory."""
    velocity = np.diff(np.asarray(obj['trans'])[:, 2]) * fps
    decreasing = np.flatnonzero(velocity < -1e-4)
    if not len(decreasing):
        raise ValueError('No lowering interval in object trajectory')
    return (float(decreasing[0] / fps), float((decreasing[-1] + 1) / fps)), dict(
        downward_speed_threshold_m_s=1e-4,
        significant_decreasing_intervals=int(len(decreasing)),
        first_last_intervals=decreasing[[0, -1]].tolist(),
        largest_upward_speed_inside_m_s=float(max(0., velocity[decreasing[0]:decreasing[-1]+1].max())),
    )


def derivative_metrics(values, times):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    dt = np.diff(times)
    velocity = np.diff(values, axis=0) / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:]) * .5)[:, None]
    return dict(
        speed_norm_magnitude=stats(np.linalg.norm(velocity, axis=1)),
        acceleration_norm_magnitude=stats(np.linalg.norm(acceleration, axis=1)),
        speed_component_rms=float(np.sqrt(np.mean(velocity * velocity))),
        acceleration_component_rms=float(np.sqrt(np.mean(acceleration * acceleration))),
    )


def angular_metrics(quat, times):
    rotations = Rotation.from_quat(quat)
    dt = np.diff(times)
    omega = (rotations[:-1].inv() * rotations[1:]).as_rotvec() / dt[:, None]
    alpha = np.diff(omega, axis=0) / ((dt[:-1] + dt[1:]) * .5)[:, None]
    return dict(
        angular_speed_rad_s=stats(np.linalg.norm(omega, axis=1)),
        angular_acceleration_rad_s2=stats(np.linalg.norm(alpha, axis=1)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--loader-samples', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--skip-collisions-on-failure', action='store_true', help='Retain a failed preliminary report without spending time on collision checks')
    args = parser.parse_args()
    # Resolve CLI paths before install() changes the working directory.
    args = argparse.Namespace(**{key: (value.resolve() if isinstance(value, Path) else value) for key, value in vars(args).items()})
    install()
    from validate_lift_hold import (URDF, array_gates, collision_assessment, independent_mujoco_fk, load_contract)
    if VIBRATION_DATA not in args.motion.parents or VIBRATION_DATA not in args.loader_samples.parents:
        raise ValueError('Motion and loader trace must stay in data_remove_vibration')
    if ROOT not in args.config.parents or ROOT not in args.report.parents:
        raise ValueError('Configuration and report must stay in converter')
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    source = Path(config['source']).resolve(strict=True)
    if VIBRATION_DATA not in source.parents:
        raise ValueError('Comparison source must be the accepted hold-stabilized reference')
    if sha(source) != config['source_sha256']:
        raise ValueError('Comparison source hash mismatch')
    path, motion, obj, urdf, joints, names, bodies = load_contract(args.motion)
    with source.open('rb') as stream:
        old = pickle.load(stream)
    with np.load(source.with_suffix('.npz'), allow_pickle=False) as z:
        old_obj = {name: z[name].copy() for name in z.files}
    immutable = {str(p): sha(p) for p in [
        source, source.with_suffix('.npz'), path, path.with_suffix('.npz'),
        args.loader_samples, args.config, URDF, ROOT / 'config/quality_gates.json',
    ]}
    fps = float(motion['fps'])
    duration = (len(motion['root_pos']) - 1) / fps
    phase, phase_detection = lowering_interval(old_obj, float(old['fps']))
    edit = np.asarray(config['edit_start_end_s'], dtype=float)
    core = np.asarray(config.get('hold_core_start_end_s', [1.4166666666666667, 4.916666666666667]), dtype=float)
    with np.load(args.loader_samples, allow_pickle=False) as z:
        if str(z['motion_sha256']) != sha(path) or z['body_names'].tolist() != bodies:
            raise ValueError('Actual-loader trace hash/body-order mismatch')
        times = z['times_s'].copy()
        sampled = {name: z[name].copy() for name in ('root_pos', 'root_rot', 'dof_pos')}
        trace_world = z['world_link_pos'].copy()
        trace_rot = z['world_link_rot'].copy()
    expected_times = np.linspace(0., duration, int(round(duration * 600)) + 1)
    if times.shape != expected_times.shape or np.max(abs(times - expected_times)) > 1e-8:
        raise ValueError('Actual-loader trace must cover the complete 600 Hz grid including exact endpoint')
    if not all(np.isfinite(a).all() for a in [times, trace_world, trace_rot] + list(sampled.values())):
        raise ValueError('Nonfinite actual-loader samples')
    if not (np.diff(times) > 0).all():
        raise ValueError('Nonmonotonic sample clock')
    array, diagnostics = array_gates(motion, obj, urdf, joints, names, bodies)
    independent = independent_mujoco_fk(motion, urdf, names, bodies)
    array['independent_mujoco_FK_10um'] = independent['max_world_position_error_m'] < 1e-5
    array['independent_mujoco_FK_rotation_1urad'] = independent['max_world_rotation_error_rad'] < 1e-6
    lower = np.array([float(j.find('limit').get('lower')) for j in joints])
    upper = np.array([float(j.find('limit').get('upper')) for j in joints])
    vmax = np.array([float(j.find('limit').get('velocity')) for j in joints])
    q = sampled['dof_pos']
    array['dense_URDF_position_limits'] = bool(((q >= lower-1e-6) & (q <= upper+1e-6)).all())
    array['dense_URDF_speed_limits'] = bool((abs(np.diff(q, axis=0))/np.diff(times)[:, None] <= vmax+1e-6).all())
    contact_gates, contact, new_fk, contact_traces = contact_assessment(sampled, urdf, names, bodies, times)
    before = pose_samples(old, times)
    old_fk = world_fk(before, urdf, names, bodies)
    trace_position_error = float(np.max(np.linalg.norm(new_fk[3]-trace_world, axis=-1)))
    expected_rot = np.stack([new_fk[2] @ new_fk[1][body] for body in bodies], axis=1)
    trace_rotation_error = float(Rotation.from_matrix(
        (np.swapaxes(expected_rot, -1, -2) @ trace_rot).reshape(-1, 3, 3)).magnitude().max())
    array['trace_world_FK_positions_10um'] = trace_position_error < 1e-5
    array['trace_world_FK_rotations_1urad'] = trace_rotation_error < 1e-6
    native_times = np.arange(len(motion['root_pos'])) / fps
    if len(old['root_pos']) != len(motion['root_pos']) or float(old['fps']) != fps:
        raise ValueError('Variant changes frame count or FPS')
    outside = (native_times <= edit[0]+1e-10) | (native_times >= edit[1]-1e-10)
    core_mask = (native_times >= core[0]-1e-10) & (native_times <= core[1]+1e-10)
    entire_hold = native_times <= float(config.get('hold_preserved_through_s', phase[0]))+1e-10
    root_displacement = np.linalg.norm(sampled['root_pos']-before['root_pos'], axis=1)
    root_angle = np.degrees(rotation_delta(sampled['root_rot'], before['root_rot']))
    limits = config.get('gates', {})
    hand = {}
    for hand_name in HANDS:
        index = bodies.index(hand_name)
        displacement = np.linalg.norm(new_fk[3][:, index]-old_fk[3][:, index], axis=1)
        angle = np.degrees(Rotation.from_matrix(
            np.swapaxes(old_fk[2]@old_fk[1][hand_name], 1, 2) @
            (new_fk[2]@new_fk[1][hand_name])).magnitude())
        hand[hand_name] = dict(position_m=stats(displacement), angle_deg=stats(angle))
    comparison_gates = dict(
        duration_fps_frame_count_exact=True,
        object_bytes_exact=sha(path.with_suffix('.npz')) == sha(source.with_suffix('.npz')),
        outside_edit_window_exact=all(np.array_equal(motion[k][outside], old[k][outside]) for k in POSE_KEYS),
        accepted_static_hold_core_exact=all(np.array_equal(motion[k][core_mask], old[k][core_mask]) for k in POSE_KEYS),
        entire_accepted_hold_exact=all(np.array_equal(motion[k][entire_hold], old[k][entire_hold]) for k in POSE_KEYS),
        root_displacement_bounded=bool(root_displacement.max() <= limits.get('maximum_root_displacement_m', .1)),
        root_orientation_bounded=bool(root_angle.max() <= limits.get('maximum_root_orientation_change_deg', 10.)),
        hand_displacement_bounded=all(row['position_m']['max'] <= limits.get('maximum_hand_displacement_m', .1) for row in hand.values()),
        hand_orientation_bounded=all(row['angle_deg']['max'] <= limits.get('maximum_hand_orientation_change_deg', 35.) for row in hand.values()),
        edit_covers_entire_lowering=bool(edit[0] <= phase[0] and edit[1] >= phase[1]),
    )
    if 'lowering_start_end_s' in config:
        comparison_gates['declared_lowering_matches_object'] = bool(np.max(abs(np.asarray(config['lowering_start_end_s'])-phase)) <= 1/fps+1e-9)
    feet = {}
    for foot in ('left_ankle_roll_link', 'right_ankle_roll_link'):
        index = bodies.index(foot)
        displacement = np.linalg.norm(new_fk[3][:, index]-old_fk[3][:, index], axis=1)
        angle = np.degrees(Rotation.from_matrix(np.swapaxes(old_fk[2]@old_fk[1][foot], 1, 2) @ (new_fk[2]@new_fk[1][foot])).magnitude())
        feet[foot] = dict(position_m=stats(displacement), angle_deg=stats(angle))
    comparison_gates['world_foot_positions_preserved_1mm'] = all(row['position_m']['max'] <= limits.get('world_foot_position_max_error_m', .001) for row in feet.values())
    comparison_gates['world_foot_rotations_preserved_0_5deg'] = all(row['angle_deg']['max'] <= limits.get('world_foot_rotation_max_error_deg', .5) for row in feet.values())
    lower_mask = (times >= phase[0]-1e-10) & (times <= phase[1]+1e-10)
    sos = butter(4, 3., 'highpass', fs=600., output='sos')
    frequency, derivatives, boundary, trend = {}, {}, {}, {}
    object_position = linear(old_obj['trans'], times, float(old['fps']))
    upper_indices = [i for i, name in enumerate(names) if any(part in name for part in ('waist_', 'shoulder_', 'elbow_', 'wrist_'))]
    leg_indices = [i for i in range(len(names)) if i not in upper_indices]
    if len(upper_indices) != 17 or len(leg_indices) != 12:
        raise ValueError('Expected 17 waist/arm joints and 12 leg joints in G1 contract')
    for label, sample, fk in [('before', before, old_fk), ('after', sampled, new_fk)]:
        channels = dict(root_position=sample['root_pos'], joint_position=sample['dof_pos'],
                        waist_and_arms=sample['dof_pos'][:, upper_indices], leg_joints=sample['dof_pos'][:, leg_indices])
        for hand_name in HANDS:
            channels[hand_name] = fk[3][:, bodies.index(hand_name)]
        # A local rotation chart adds a root angular vibration diagnostic.
        channels['root_rotation_chart_rad'] = (Rotation.from_quat(sample['root_rot'][0]).inv() *
                                               Rotation.from_quat(sample['root_rot'])).as_rotvec()
        frequency[label] = {}
        derivatives[label] = {}
        for name, values in channels.items():
            highpass = sosfiltfilt(sos, values, axis=0)
            frequency[label][name] = float(np.sqrt(np.mean(highpass[lower_mask] ** 2)))
            derivatives[label][name] = derivative_metrics(values[lower_mask], times[lower_mask])
        derivatives[label]['root_angular'] = angular_metrics(sample['root_rot'][lower_mask], times[lower_mask])
        boundary[label] = {}
        for center in sorted(set(edit.tolist() + list(phase))):
            mask = (times >= center-.1-1e-10) & (times <= center+.1+1e-10)
            boundary[label][str(center)] = {name: derivative_metrics(values[mask], times[mask]) for name, values in channels.items()}
            boundary[label][str(center)]['root_angular'] = angular_metrics(sample['root_rot'][mask], times[mask])
        trend[label] = {}
        for hand_name in HANDS:
            relative = channels[hand_name] - object_position
            first, last = relative[lower_mask][0], relative[lower_mask][-1]
            trend[label][hand_name] = dict(
                relative_to_object_first_xyz_m=first.tolist(), relative_to_object_last_xyz_m=last.tolist(),
                relative_net_change_xyz_m=(last-first).tolist(),
                relative_xyz_range_m=np.ptp(relative[lower_mask], axis=0).tolist(),
                world_hand_net_change_xyz_m=(channels[hand_name][lower_mask][-1]-channels[hand_name][lower_mask][0]).tolist(),
                object_net_change_xyz_m=(object_position[lower_mask][-1]-object_position[lower_mask][0]).tolist(),
            )
    ratios = {name: float(frequency['after'][name] / max(frequency['before'][name], 1e-15)) for name in frequency['before']}
    default_channels = ['root_position', 'joint_position'] + list(HANDS)
    gate_channels = config.get('high_frequency_gate_channels', default_channels)
    if not isinstance(gate_channels, list) or not gate_channels or len(set(gate_channels)) != len(gate_channels) or any(name not in ratios for name in gate_channels):
        raise ValueError('Invalid explicit high-frequency gate channel selection')
    if gate_channels != default_channels and not config.get('high_frequency_profile_explanation'):
        raise ValueError('A changed frequency profile requires an explicit explanation in configuration')
    vibration_gates = {name+'_HF_RMS_reduction_at_least_80percent': bool(ratios[name] <= limits.get('high_frequency_rms_ratio_max', .2))
                       for name in gate_channels}
    if 'leg_high_frequency_rms_ratio_max' in limits:
        vibration_gates['leg_joint_HF_RMS_no_material_increase'] = bool(ratios['leg_joints'] <= limits['leg_high_frequency_rms_ratio_max'])
    per_leg_joint = {}
    before_joint_hf = sosfiltfilt(sos, before['dof_pos'], axis=0)
    after_joint_hf = sosfiltfilt(sos, sampled['dof_pos'], axis=0)
    lower_indices = np.flatnonzero(lower_mask)
    for index in leg_indices:
        name = names[index]
        side = 0 if name.startswith('left_') else 1
        foot = 'left_ankle_roll_link' if side == 0 else 'right_ankle_roll_link'
        cyl = contact_traces['cylinder_gap'][:, side]
        cap = contact_traces['capsule_gap'][:, side]
        near_support = np.maximum(abs(cyl), abs(cap)) <= .005
        old_hf = before_joint_hf[:, index]
        new_hf = after_joint_hf[:, index]
        old_rms = float(np.sqrt(np.mean(old_hf[lower_mask]**2)))
        new_rms = float(np.sqrt(np.mean(new_hf[lower_mask]**2)))
        peaks = lower_indices[np.argsort(-abs(new_hf[lower_mask]))[:3]]
        energy = new_hf[lower_mask]**2
        total_energy = float(energy.sum())
        per_leg_joint[name] = dict(before_HF_RMS_rad=old_rms, after_HF_RMS_rad=new_rms,
            after_over_before=float(new_rms/max(old_rms,1e-15)), delta_HF_RMS_rad=new_rms-old_rms,
            delta_HF_RMS_deg=float(np.degrees(new_rms-old_rms)),
            before_derivatives=derivative_metrics(before['dof_pos'][lower_mask,index],times[lower_mask]),
            after_derivatives=derivative_metrics(sampled['dof_pos'][lower_mask,index],times[lower_mask]),
            after_HF_energy_fraction_foot_within_support_band=float(energy[near_support[lower_mask]].sum()/max(total_energy,1e-30)),
            after_HF_energy_fraction_foot_outside_support_band=float(energy[~near_support[lower_mask]].sum()/max(total_energy,1e-30)),
            largest_after_HF_samples=[dict(time_s=float(times[k]), HF_rad=float(new_hf[k]),
                joint_position_rad=float(sampled['dof_pos'][k,index]),
                foot_world_z_m=float(new_fk[3][k,bodies.index(foot),2]),
                foot_native_gap_m=float(cyl[k]), foot_capsule_gap_m=float(cap[k]),
                foot_within_support_band=bool(near_support[k])) for k in peaks])
    all_initial = {**array, **contact_gates, **comparison_gates, **vibration_gates}
    validation_sources = [Path(__file__), Path(__file__).with_name('validate_lift_hold.py'),
                          Path(__file__).with_name('lift_hold_validation_common.py'),
                          Path(__file__).with_name('lift_hold_collision_oracle.py'),
                          Path(__file__).with_name('test_collision_aggregation.py')]
    report = dict(
        status='VALIDATING_COLLISIONS', motion=str(path), motion_sha256=sha(path), source=str(source), source_sha256=sha(source),
        configuration_path=str(args.config), configuration_sha256=sha(args.config), configuration=config,
        actual_loader_samples=str(args.loader_samples), actual_loader_samples_sha256=sha(args.loader_samples),
        frames=len(native_times), fps=fps, duration_s=duration, sample_count=len(times), sample_rate_hz=600,
        lowering_start_end_s=list(phase), lowering_phase_detection=phase_detection,
        array_gates=array, contact_gates=contact_gates, comparison_gates=comparison_gates, vibration_gates=vibration_gates,
        diagnostics=diagnostics, independent_mujoco_fk=independent, contact=contact,
        trace_FK_error=dict(position_m=trace_position_error, rotation_rad=trace_rotation_error),
        comparison=dict(root_position_m=stats(root_displacement), root_angle_deg=stats(root_angle), hands=hand, feet=feet),
        high_frequency_RMS_above_3Hz=frequency, high_frequency_after_over_before=ratios,
        high_frequency_gate_channels=gate_channels, high_frequency_joint_groups=dict(waist_and_arms=[names[i] for i in upper_indices], legs=[names[i] for i in leg_indices]),
        lowering_derivatives=derivatives, boundary_100ms_windows=boundary, hand_object_relative_trend=trend,
        per_leg_joint_HF_diagnostics=per_leg_joint,
        per_leg_joint_localization_note='Foot within support band uses both native and capsule gaps within5mm. Outside this band is a kinematic airborne/transition diagnostic, not a measured contact label. Highpass energy can spread across phase boundaries.',
        high_frequency_definition='Fourth-order Butterworth highpass at 3Hz, zero-phase over entire clip; RMS evaluated only during independently inferred lowering. Root rotation uses a local rotation-vector chart.',
        units='Position channels in m, joint/root rotation channels in rad; derivatives per s or s².',
        validation_source_sha256={str(p): sha(p) for p in validation_sources},
        policy_executed=False, training_executed=False, physics_executed=False,
        limitations=[
            'Finite sampled geometry is not a continuous-time proof, balance or grasp-force certification.',
            'Outside-window source motion is deliberately retained, including its velocity variation.',
            'Boundary acceleration includes piecewise interpolation knots and float32 loader time quantization.',
            'Hand/object-relative trends are diagnostics of reference changes, not measured physical grasp constraints.',
        ],
    )
    write_json(args.report, report)
    if args.skip_collisions_on_failure and not all(all_initial.values()):
        collision_gates = {'full_collision_coverage_completed': False}
        collision = dict(status='NOT_TESTED_PRELIMINARY_FAILURE', reason='Initial validation gates failed; no complete reference PASS is possible')
    else:
        collision_gates, collision = collision_assessment(
            motion, urdf, names, bodies, 600., args.loader_samples, sha(path))
    report.update(collision_gates=collision_gates, collision=collision)
    preservation = {str(p): sha(p) == h for p, h in immutable.items()}
    report['input_hashes_preserved'] = preservation
    all_gates = {**all_initial, **collision_gates, 'all_inputs_preserved': all(preservation.values())}
    report['failed_gates'] = [key for key, value in all_gates.items() if not value]
    report['gate_count'] = len(all_gates)
    report['status'] = 'PASS_LOWERING_REFERENCE' if not report['failed_gates'] else 'FAIL_LOWERING_REFERENCE'
    report['elapsed_s'] = time.monotonic()-started
    write_json(args.report, report)
    print(json.dumps(dict(status=report['status'], failed_gates=report['failed_gates'],
                          high_frequency_after_over_before=ratios, report=str(args.report))))
    if report['failed_gates']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
