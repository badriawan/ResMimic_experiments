"""Read-only input inspection for a conversion PLAN, never a converter.

Run with the existing gmr_omomo Python, -I -B. The trusted local InterPose
project NPY is a pickled dictionary; numeric NPZ archives use allow_pickle=False.
All newly written reports are constrained to this converter directory.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'lift_hold_001'
REPO = ROOT.parents[2]
EPISODE = Path('output/frl_apartment_4_sub17_floorlamp_001_0')
MOTION = EPISODE / 'sub17_floorlamp_001_pidx_0_task_0.npy'
RECORDER = EPISODE / 'maskedmimic_res/task_0/stage10_raw_trajectory.npz'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, data):
    path = (ROOT / 'audit' / name).resolve()
    if ROOT not in path.parents:
        raise ValueError('Report outside converter directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def describe(value):
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        data = dict(shape=list(value.shape), dtype=str(value.dtype))
        if value.dtype.kind in 'fiu' and value.size:
            data.update(finite=bool(np.isfinite(value).all()), min=float(value.min()), max=float(value.max()))
        if value.size <= 32 and value.dtype.kind != 'O':
            data['values'] = value.tolist()
        elif value.size and value.dtype.kind != 'O':
            data['first_values'] = value.reshape(-1)[:6].tolist()
            data['last_values'] = value.reshape(-1)[-6:].tolist()
        return data
    if isinstance(value, (tuple, list)):
        if len(value) <= 32:
            return [describe(v) for v in value]
        return dict(length=len(value), first=[describe(v) for v in value[:2]], last=[describe(value[-1])])
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return dict(type=type(value).__name__)


def main():
    generated = SOURCE / 'data_after_conversion'
    files = sorted(p for p in SOURCE.rglob('*') if p.is_file() and generated not in p.parents and (SOURCE/'data_remove_vibration') not in p.parents)
    records = [dict(path=str(p.relative_to(SOURCE)), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    hashes = {v['path']: v['sha256'] for v in records}
    bundle = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    write('source_inventory.json', dict(source=str(SOURCE), files=len(files),
        total_bytes=sum(v['bytes'] for v in records), extensions=dict(Counter(p.suffix for p in files)),
        bundle_inventory_sha256=bundle, hash_scope='every copied source file', records=records))

    recovered = json.loads((SOURCE / 'resume_manifest.json').read_text())
    verified = []
    for old_path, expected in recovered['artifacts']['sha256'].items():
        relative = old_path.split('/runs/lift_hold_001/', 1)[1]
        verified.append(dict(path=relative, expected=expected, actual=hashes.get(relative),
                             matches=hashes.get(relative) == expected))
    # This is the user's trusted generated project dictionary, not the seed.
    project = np.load(SOURCE / MOTION, allow_pickle=True).item()
    schema = dict(primary_motion=str(MOTION), project=describe(project), numeric_archives={})
    for path in sorted(p for p in SOURCE.rglob('*.npz') if generated not in p.parents and (SOURCE/'data_remove_vibration') not in p.parents):
        fields = {}
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                try:
                    fields[key] = describe(z[key])
                except ValueError:
                    fields[key] = dict(object_array_not_loaded=True)
        schema['numeric_archives'][str(path.relative_to(SOURCE))] = fields
    write('source_schema.json', schema)

    human = project['human']['joint_xyz'][0]
    obj = project['object']['floorlamp']['xyz'][0]
    quat = project['object']['floorlamp']['rotation'][0]
    pelvis = human[:, 0]
    fps = float(project['metadata']['fps'])
    with np.load(SOURCE / RECORDER, allow_pickle=False) as z:
        ht = z['recorder_reference_time_s'].copy()
        st = z['recorder_sdk_time_s'].copy()
        raw_root = z['root_pos'].copy()
    ot = np.arange(len(obj), dtype=np.float64) / fps
    peak = np.flatnonzero(np.isclose(obj[:, 2], obj[:, 2].max(), atol=1e-6, rtol=0))
    common_first = int(np.ceil((max(ht[0], ot[0]) - 1e-5) * fps))
    common_last = int(np.floor((min(ht[-1], ot[-1]) + 1e-5) * fps))
    provisional_samples = common_last - common_first + 1
    findings = dict(status='INPUT_INSPECTION_ONLY', created_utc=datetime.now(timezone.utc).isoformat(),
        conversion_executed=False, retarget_executed=False, policy_executed=False, training_executed=False,
        source=str(SOURCE), output_root=str(ROOT), copied_artifact_hash_checks=verified,
        original_manifest_status=json.loads((SOURCE / 'manifest.json').read_text())['status'],
        recovered_manifest_status=recovered['status'],
        frames=len(human), fps=fps, source_endpoint_duration_if_index_clock_s=(len(human)-1)/fps,
        source_video_frame_duration_s=len(human)/fps,
        human_reference_clock_first_last_s=ht[[0,-1]].tolist(),
        human_sdk_clock_first_last_s=st[[0,-1]].tolist(),
        derived_object_clock_first_last_s=ot[[0,-1]].tolist(),
        reference_clock_strictly_increasing=bool((np.diff(ht)>0).all()),
        clock_evidence=project['metadata']['runtime_clock_observation']['clock_validation'],
        provisional_common_grid=dict(status='CALCULATED_NOT_EXPORTED',first_k=common_first,last_k=common_last,
            frames_at_30Hz=provisional_samples, endpoint_duration_s=(provisional_samples-1)/fps,
            condition='Native30Hz supported overlap with rounding tolerance1e-5s; final decision requires phase/end review'),
        dense_object=dict(first_xyz_m=obj[0].tolist(),last_xyz_m=obj[-1].tolist(),
            peak_lift_above_first_m=float(obj[:,2].max()-obj[0,2]),
            max_xy_displacement_m=float(np.linalg.norm(obj[:,:2]-obj[0,:2],axis=1).max()),
            peak_frame_first_last=[int(peak[0]),int(peak[-1])],
            peak_plateau_contiguous=bool((np.diff(peak)==1).all()),
            peak_plateau_duration_s=(int(peak[-1])-int(peak[0]))/fps if bool((np.diff(peak)==1).all()) else None,
            max_quaternion_norm_error=float(np.abs(np.linalg.norm(quat,axis=-1)-1).max()),
            scope='Stored reference trajectory; no physical lifting validation'),
        human_pelvis=dict(first_xyz_m=pelvis[0].tolist(),last_xyz_m=pelvis[-1].tolist(),
            max_xy_displacement_from_first_m=float(np.linalg.norm(pelvis[:,:2]-pelvis[0,:2],axis=1).max()),
            stationary_verified=False),
        raw_root=dict(first=raw_root[0].tolist(),last=raw_root[-1].tolist(),min=float(raw_root.min()),max=float(raw_root.max()),
            note='Recorder/SDK coordinates; do not replace scene-normalized primary human root without frame audit'),
        goal=project['metadata']['custom_goal'],
        source_physical_success_verified=project['metadata']['physical_success_verified'],
        source_phase_sync_verified=project['metadata']['phase_sync_verified'],
        source_resmimic_compatibility_verified=project['metadata']['resmimic_compatibility_verified'],
        sidecar_caution='trajectory_pose_aa legacy neutral/10betas differs from primary SMPL-X male/16betas; use primary joint_xyz +22localAA',
        limitations=['No IK, alignment export, correction, simulator, policy, or training was run.',
            'Body drift and object clock/contact need phase review; artifact hashes do not certify physical success.'])
    write('findings.json', findings)

    dependency_paths = dict(
        target_urdf=REPO/'assets/g1/g1_custom_collision_29dof.urdf',
        gmr_xml=REPO/'GMR/assets/unitree_g1/g1_mocap_29dof.xml',
        gmr_config=REPO/'GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json',
        actual_robot_loader=REPO/'pose/pose/utils/motion_lib_pkl.py',
        actual_hoi_loader=REPO/'pose/pose/utils/motion_lib_hoi.py',
        base_policy=REPO/'legged_gym/assets/checkpoints/base_policy.pt',
        floorlamp_rest_candidate=Path('/home/badriawan/InterPose/InterPose/processed_data/rest_object_geo/floorlamp.ply'))
    dependencies = {name: dict(path=str(p), exists=p.is_file(), sha256=sha(p) if p.is_file() else None)
                    for name,p in dependency_paths.items()}
    urdf = ET.parse(dependency_paths['target_urdf']).getroot()
    xml = ET.parse(dependency_paths['gmr_xml']).getroot()
    target = dict(units=dict(position='meter',joint_angles='radian',up_axis='Z'),
        pkl_quaternion_order='XYZW',object_npz_quaternion_order='XYZW',
        qpos_quaternion_order='WXYZ',
        dof_order=[j.get('name') for j in urdf.findall('joint') if j.get('type')!='fixed'],
        body_order=[b.get('name') for w in xml.findall('worldbody') for b in w.iter('body')],
        base_actor_observations=1757,actions=29)
    write('dependencies_and_target.json', dict(dependencies=dependencies,target_contract=target,
        inspection_python=sys.executable,numpy=np.__version__,source_modified=False))
    print(json.dumps(dict(status='INSPECTED_NOT_CONVERTED', files=len(files), all_recovery_hashes_match=all(v['matches'] for v in verified),
                         frames=len(human),fps=fps,report_root=str(ROOT/'audit'))))


if __name__ == '__main__':
    main()
