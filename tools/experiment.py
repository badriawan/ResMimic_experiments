#!/usr/bin/env python3
"""Portable InterPose geometry experiments; no policy training or physics control.

Run through tools/run_geometry.sh, or select a Python 3.11 geometry environment.
The supported installation layout is ResMimic/InterPose/custom_data.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT / "configs/datasets.json"
NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
         "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
         "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def resmimic_root():
    return Path(os.environ.get("RESMIMIC_ROOT", str(PROJECT.parents[1]))).expanduser().resolve()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def show(data):
    print(json.dumps(data, indent=2, allow_nan=False), flush=True)


def write_json(path, data):
    path = Path(path)
    with path.open("x") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def project_file(value):
    path = (PROJECT / value).resolve()
    if PROJECT not in path.parents:
        raise ValueError("Configured input must be inside this repository: " + str(value))
    return path


def read_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    return path.resolve(strict=True)


def output_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    allowed = (PROJECT / "outputs").resolve()
    if path != allowed and allowed not in path.parents:
        raise ValueError("New outputs must stay under " + str(allowed))
    return path


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def new_files(paths):
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing artifacts: " + ", ".join(existing))


def config(name):
    datasets = json.loads(DATASETS.read_text())["datasets"]
    if name not in datasets:
        raise ValueError("Unknown dataset: " + name)
    value = dict(datasets[name])
    for key in ("source_npy", "clock_npz", "rest_mesh"):
        value[key] = project_file(value[key])
    if value["origin_policy"] not in ("aligned_first", "source_first"):
        raise ValueError("Unsupported origin policy")
    if not 0 < float(value["scale"]) <= 2:
        raise ValueError("Invalid scene scale")
    return value


def geometry():
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError as error:
        raise RuntimeError("Geometry dependencies are missing. Install requirements/geometry.txt and set GMR_PYTHON.") from error
    return np, Rotation, Slerp


def configure_runtime():
    """Keep geometry caches and temporary files in the experiment repository."""
    runtime = PROJECT / ".runtime/geometry"
    for key, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"), ("MPLCONFIGDIR", "matplotlib"),
                      ("__GL_SHADER_DISK_CACHE_PATH", "nvidia"), ("MESA_SHADER_CACHE_DIR", "mesa")):
        path = runtime / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    sys.dont_write_bytecode = True


def doctor():
    repo = resmimic_root()
    checks = {
        "dataset_config": DATASETS.is_file(),
        "resmimic_g1_urdf": (repo / "assets/g1/g1_custom_collision_29dof.urdf").is_file(),
        "resmimic_fk_helper": (repo / "legged_gym/legged_gym/scripts/convert_gmr_qpos.py").is_file(),
        "resmimic_reference_renderer": (repo / "legged_gym/legged_gym/scripts/render_motion.py").is_file(),
        "gmr_robot_xml": (repo / "GMR/assets/unitree_g1/g1_mocap_29dof.xml").is_file(),
        "gmr_ik_config": (repo / "GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json").is_file(),
    }
    pins_path = PROJECT / "configs/dependency_pins.json"
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}
    revisions = {}
    for key, path in (("resmimic", repo), ("gmr", repo / "GMR")):
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        revisions[key] = result.stdout.strip() if result.returncode == 0 else None
        if pins.get(key, {}).get("commit"):
            checks[key + "_pinned_commit"] = revisions[key] == pins[key]["commit"]
    if DATASETS.exists():
        for name in json.loads(DATASETS.read_text())["datasets"]:
            cfg = config(name)
            for key in ("source_npy", "clock_npz", "rest_mesh"):
                checks[name + "_" + key] = cfg[key].is_file()
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "mujoco", "mink", "trimesh", "imageio", "PIL")}
    report = dict(status="PASS_PATH_CHECKS" if all(checks.values()) else "MISSING_PREREQUISITES", checks=checks,
                  python=sys.executable, python_version=sys.version.split()[0], recommended_geometry_python="3.11",
                  resmimic_root=str(repo), revisions=revisions, dependency_modules=dependencies,
                  note="Path/installation checks only; no simulator, IK, policy, or optimizer runs.")
    show(report)
    return 0 if all(checks.values()) else 1


def load_source(cfg):
    np, _, _ = geometry()
    # Only load the trusted generated dictionaries supplied with this experiment.
    raw = np.load(cfg["source_npy"], allow_pickle=True).item()
    meta = raw["metadata"]
    if meta.get("position_units") != "meter" or meta.get("axis_angle_units") != "radian" or meta.get("quaternion_order", "").lower() != "wxyz":
        raise ValueError("Source must declare meter/radian/WXYZ conventions")
    human = np.asarray(raw["human"]["joint_xyz"])[0]
    aa = np.asarray(raw["human"]["pose_aa"])[0]
    obj = raw["object"]["floorlamp"]
    positions = np.asarray(obj["xyz"])[0]
    quaternions = np.asarray(obj["rotation"])[0]
    count = len(human)
    if human.shape != (count, 24, 3) or aa.shape != (count, 22, 3) or positions.shape != (count, 3) or quaternions.shape != (count, 4):
        raise ValueError("Unexpected source human/object array shapes")
    if not all(np.isfinite(v).all() for v in (human, aa, positions, quaternions)):
        raise ValueError("Nonfinite source data")
    if list(meta["pose_joint_names"]) != NAMES or list(meta["pose_joint_parent_indices"]) != PARENTS:
        raise ValueError("Source hierarchy differs from the validated SMPL-X22 order")
    fps = float(meta["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source FPS")
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1., rtol=0., atol=1e-5):
        raise ValueError("Source object quaternions are not unit quaternions")
    return raw, human, aa, positions, quaternions, fps


def inspect_source(name):
    cfg = config(name)
    raw, human, aa, positions, quaternions, fps = load_source(cfg)
    show(dict(dataset=name, frames=len(human), fps=fps, source_npy=relative(cfg["source_npy"]),
              source_sha256=sha(cfg["source_npy"]), human_joint_xyz_shape=list(human.shape),
              human_local_axis_angle_shape=list(aa.shape), object_position_shape=list(positions.shape),
              object_rotation_shape=list(quaternions.shape), quaternion_order="WXYZ", scale=cfg["scale"],
              origin_policy=cfg["origin_policy"], clock_kind=cfg["clock_kind"],
              pose_joint_names=raw["metadata"]["pose_joint_names"]))


def linear(values, source_time, query):
    np, _, _ = geometry()
    shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    return np.stack([np.interp(query, source_time, flat[:, column]) for column in range(flat.shape[1])], axis=-1).reshape((len(query),) + shape)


def common_clock(human_time, object_time, fps, tolerance=1e-5):
    np, _, _ = geometry()
    if not np.isfinite(human_time).all() or not np.isfinite(object_time).all() or not (np.diff(human_time) > 0).all() or not (np.diff(object_time) > 0).all():
        raise ValueError("Human and object clocks must be finite and strictly increasing")
    low, high = max(human_time[0], object_time[0]), min(human_time[-1], object_time[-1])
    first = int(np.ceil((low - tolerance) * fps))
    last = int(np.floor((high + tolerance) * fps))
    times = np.arange(first, last + 1, dtype=np.float64) / fps
    if len(times) < 2:
        raise ValueError("Insufficient common clock coverage")
    qh = np.clip(times, human_time[0], human_time[-1])
    qo = np.clip(times, object_time[0], object_time[-1])
    if max(float(abs(qh - times).max()), float(abs(qo - times).max())) >= tolerance:
        raise ValueError("Common clock would require extrapolation beyond roundoff")
    return times, qh, qo


def align(name, output):
    np, Rotation, Slerp = geometry()
    cfg = config(name)
    _, human, aa, positions, quaternions, fps = load_source(cfg)
    out = output_path(output)
    paths = [out / "aligned.npz", out / "alignment_report.json"]
    new_files(paths)
    with np.load(cfg["clock_npz"], allow_pickle=False) as data:
        if cfg["clock_kind"] == "pick_place_clock_map":
            ht = data["measured_human_reference_time_s"].copy()
            ot = data["derived_object_interpolation_time_s"].copy()
            if not np.array_equal(data["raw_recorder_index"], np.arange(len(human))):
                raise ValueError("Source clock map skips or reorders source frames")
        elif cfg["clock_kind"] == "lift_hold_recorder":
            ht = data["recorder_reference_time_s"].copy()
            ot = np.arange(len(positions), dtype=float) / fps
        else:
            raise ValueError("Unsupported clock_kind")
    if len(ht) != len(human) or len(ot) != len(positions):
        raise ValueError("Clock and source array lengths differ")
    times, qh, qo = common_clock(ht, ot, fps)
    hp = linear(human, ht, qh)
    haa = np.stack([Slerp(ht, Rotation.from_rotvec(aa[:, joint]))(qh).as_rotvec() for joint in range(22)], axis=1)
    op = linear(positions, ot, qo)
    oq = Slerp(ot, Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]))(qo).as_quat()
    # Normalize quaternion signs sequentially without changing physical rotations.
    for index in range(1, len(oq)):
        if np.dot(oq[index - 1], oq[index]) < 0:
            oq[index] *= -1
    origin = np.zeros(3)
    origin[:2] = (hp[0, 0] if cfg["origin_policy"] == "aligned_first" else human[0, 0])[:2]
    out.mkdir(parents=True, exist_ok=True)
    with paths[0].open("xb") as stream:
        np.savez_compressed(stream, human_joint_pos=hp, human_local_axis_angle=haa, object_pos=op,
                            object_quat_wxyz=oq[:, [3, 0, 1, 2]], fps=np.asarray(fps), source_time_s=times,
                            joint_parents22=np.asarray(PARENTS), origin_xyz=origin)
    report = dict(status="PASS_NUMERIC_ALIGNMENT", dataset=name, source_frames=len(human), frames=len(times), fps=fps,
                  duration_endpoint_s=(len(times) - 1) / fps, source_time_first_last_s=times[[0, -1]].tolist(),
                  source_human_first_last_s=ht[[0, -1]].tolist(), source_object_first_last_s=ot[[0, -1]].tolist(),
                  max_human_boundary_clamp_s=float(abs(qh - times).max()), max_object_boundary_clamp_s=float(abs(qo - times).max()),
                  extrapolation_used=False, physical_synchronization_verified=False,
                  origin_policy=cfg["origin_policy"], origin_xyz_m=origin.tolist(),
                  input_sha256={relative(cfg[key]): sha(cfg[key]) for key in ("source_npy", "clock_npz")},
                  output=relative(paths[0]), output_sha256=sha(paths[0]),
                  note="Common sampled time only. Clock agreement does not certify physical grasp/contact synchronization.")
    write_json(paths[1], report)
    show(report)


def fk_helpers():
    path = resmimic_root() / "legged_gym/legged_gym/scripts"
    if not (path / "convert_gmr_qpos.py").is_file():
        raise FileNotFoundError("Pinned ResMimic FK helper missing; run doctor and follow docs/INSTALL.md")
    sys.path.insert(0, str(path))
    from convert_gmr_qpos import forward_kinematics, foot_clearance, clearance_summary
    return forward_kinematics, foot_clearance, clearance_summary


def joint_limit_report(urdf, dof, names, fps):
    np, _, _ = geometry()
    by_name = {joint.get("name"): joint for joint in urdf.findall("joint") if joint.get("type") != "fixed"}
    rows = []
    for column, name in enumerate(names):
        limits = by_name[name].find("limit")
        low, high, speed = [float(limits.get(key)) for key in ("lower", "upper", "velocity")]
        values = dof[:, column]
        velocity = abs(np.diff(values)) * fps
        rows.append(dict(joint=name, minimum_rad=float(values.min()), maximum_rad=float(values.max()),
                         lower_rad=low, upper_rad=high, velocity_limit_rad_s=speed,
                         maximum_velocity_rad_s=float(velocity.max()),
                         position_violations=int(((values < low - 1e-6) | (values > high + 1e-6)).sum()),
                         velocity_violations=int((velocity > speed + 1e-6).sum())))
    return rows


def retarget(name, output):
    np, Rotation, _ = geometry()
    import trimesh
    cfg = config(name)
    repo = resmimic_root()
    out = output_path(output)
    aligned = out / "aligned.npz"
    if not aligned.is_file():
        raise FileNotFoundError("Run align with the same --dataset and --output first")
    alignment_report = json.loads((out / "alignment_report.json").read_text())
    if alignment_report["dataset"] != name or alignment_report["output_sha256"] != sha(aligned):
        raise ValueError("Alignment dataset/hash does not match the requested retargeting input")
    outputs = [out / name for name in ("motion.pkl", "motion.npz", "qpos.npy", "floorlamp.obj", "retarget_report.json", "uniform_ik.json")]
    new_files(outputs)
    with np.load(aligned, allow_pickle=False) as arrays:
        data = {key: arrays[key].copy() for key in arrays.files}
    positions = data["human_joint_pos"]
    aa = data["human_local_axis_angle"]
    origin = data["origin_xyz"]
    fps = float(data["fps"])
    count = len(positions)
    rotations = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(count, 22, 3, 3)
    for joint in range(1, 22):
        rotations[:, joint] = rotations[:, PARENTS[joint]] @ rotations[:, joint]
    quats = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(count, 22, 4)
    sys.path.insert(0, str(repo / "GMR"))
    try:
        from general_motion_retargeting import GeneralMotionRetargeting, IK_CONFIG_DICT
    except ImportError as error:
        raise RuntimeError("GMR/Mink dependencies missing; follow docs/INSTALL.md and set GMR_PYTHON") from error
    config_source = repo / "GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json"
    ik = json.loads(config_source.read_text())
    scale = float(cfg["scale"])
    ik["human_scale_table"] = {key: scale for key in ik["human_scale_table"]}
    write_json(out / "uniform_ik.json", ik)
    label = "custom_data_" + name
    IK_CONFIG_DICT[label] = {"unitree_g1": str(out / "uniform_ik.json")}
    solver = GeneralMotionRetargeting(label, "unitree_g1", actual_human_height=None, use_velocity_limit=True, verbose=False)
    if set(solver.human_scale_table.values()) != {scale}:
        raise ValueError("GMR did not retain the requested uniform scene scale")
    def frame(index):
        return {name: (positions[index, joint].copy() - origin, quats[index, joint].copy()) for joint, name in enumerate(NAMES)}
    started = time.monotonic()
    for _ in range(10):
        solver.retarget(frame(0))
    qpos = []
    task_residuals = {}
    for index in range(count):
        qpos.append(solver.retarget(frame(index)))
        for stage, tasks in (("stage1", solver.human_body_to_task1), ("stage2", solver.human_body_to_task2)):
            for body, task in tasks.items():
                task_residuals.setdefault(stage + "." + body, []).append(float(np.linalg.norm(task.compute_error(solver.configuration)[:3])))
        if index % 60 == 0 or index == count - 1:
            print("GMR retarget frame %d/%d" % (index + 1, count), flush=True)
    qpos = np.asarray(qpos)
    if qpos.shape != (count, 36) or not np.isfinite(qpos).all() or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., rtol=0., atol=1e-6):
        raise ValueError("Expected finite GMR qpos[T,36] with WXYZ root quaternion")
    xml_path = repo / "GMR/assets/unitree_g1/g1_mocap_29dof.xml"
    urdf_path = repo / "assets/g1/g1_custom_collision_29dof.urdf"
    xml, urdf = ET.parse(xml_path).getroot(), ET.parse(urdf_path).getroot()
    source_names = [joint.get("name") for joint in xml.findall(".//body/joint")]
    names = [joint.get("name") for joint in urdf.findall("joint") if joint.get("type") != "fixed"]
    bodies = [body.get("name") for world in xml.findall("worldbody") for body in world.iter("body")]
    if len(names) != 29 or len(bodies) != 38 or set(names) != set(source_names):
        raise ValueError("Pinned source and target G1 joint sets do not match")
    dof = qpos[:, 7:][:, [source_names.index(joint) for joint in names]]
    limits = joint_limit_report(urdf, dof, names, fps)
    forward_kinematics, foot_clearance, clearance_summary = fk_helpers()
    local_pos, local_rot = forward_kinematics(urdf, dof, names, bodies)
    root_xyzw = qpos[:, [4, 5, 6, 3]].copy()
    root_xyzw /= np.linalg.norm(root_xyzw, axis=1, keepdims=True)
    motion = dict(fps=fps, root_pos=qpos[:, :3].copy(), root_rot=root_xyzw, dof_pos=dof, local_body_pos=local_pos, link_body_list=bodies)
    object_pos = scale * (data["object_pos"] - origin)
    object_rot = data["object_quat_wxyz"][:, [1, 2, 3, 0]].copy()
    object_rot /= np.linalg.norm(object_rot, axis=1, keepdims=True)
    mesh = trimesh.load_mesh(cfg["rest_mesh"], process=False)
    mesh.vertices = np.asarray(mesh.vertices) * scale
    violation_count = sum(row["position_violations"] + row["velocity_violations"] for row in limits)
    report = dict(status="PASS_KINEMATIC_EXPORT" if not violation_count else "CANDIDATE_WITH_JOINT_LIMIT_VIOLATIONS", dataset=name,
                  frames=count, fps=fps, duration_endpoint_s=(count - 1) / fps, scene_uniform_scale=scale,
                  source_origin_xyz_m=origin.tolist(), transform="p_out=scale*(p_source-origin_xy); no extra floor shift or recentering",
                  root_and_object_quaternion="XYZW", qpos_quaternion="WXYZ", joint_order=names, link_body_list=bodies,
                  joint_limits=limits, joint_violation_count=violation_count,
                  ik_position_task_residual_m={key: dict(maximum=float(max(values)), mean=float(np.mean(values))) for key, values in task_residuals.items()},
                  feet=clearance_summary(foot_clearance(urdf, local_pos, local_rot, bodies, motion["root_pos"], Rotation.from_quat(root_xyzw).as_matrix())),
                  input_sha256={relative(path): sha(path) for path in (aligned, cfg["rest_mesh"], config_source, xml_path, urdf_path)},
                  elapsed_s=time.monotonic() - started, physical_success_verified=False, policy_or_training_run=False,
                  note="Actual GMR/Mink IK and URDF FK. Exported candidates are never silently clipped. Geometry, force closure and dynamic balance require subsequent validation.")
    with (out / "qpos.npy").open("xb") as stream:
        np.save(stream, qpos)
    with (out / "motion.pkl").open("xb") as stream:
        pickle.dump(motion, stream, protocol=4)
    with (out / "motion.npz").open("xb") as stream:
        np.savez_compressed(stream, trans=object_pos, rot=object_rot, source_time_s=data["source_time_s"])
    mesh.export(out / "floorlamp.obj")
    report["output_sha256"] = {relative(path): sha(path) for path in outputs if path.exists()}
    write_json(out / "retarget_report.json", report)
    show(report)
    return 0 if violation_count == 0 else 2


class CompatibleUnpickler(pickle.Unpickler):
    """Read NumPy2 recordings when inspecting them in the NumPy1 SDK environment."""
    def find_class(self, module, name):
        if importlib.util.find_spec("numpy._core") is None:
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_motion(path):
    np, _, _ = geometry()
    with path.open("rb") as stream:
        motion = CompatibleUnpickler(stream).load()
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        objects = {key: arrays[key].copy() for key in arrays.files}
    return motion, objects


def validate_motion(path, report_path=None):
    np, Rotation, _ = geometry()
    path = read_path(path)
    motion, objects = load_motion(path)
    frames = len(motion["root_pos"])
    fps = float(motion["fps"])
    shapes = {"root_pos": (frames, 3), "root_rot": (frames, 4), "dof_pos": (frames, 29), "local_body_pos": (frames, 38, 3)}
    checks = {"frames_at_least_two": frames >= 2, "positive_fps": bool(np.isfinite(fps) and fps > 0),
              "robot_shapes": all(np.shape(motion.get(key)) == shape for key, shape in shapes.items()),
              "body_names": len(motion.get("link_body_list", [])) == 38 and len(set(motion.get("link_body_list", []))) == 38,
              "object_shapes": objects.get("trans", np.empty(0)).shape == (frames, 3) and objects.get("rot", np.empty(0)).shape == (frames, 4)}
    if not all(checks.values()):
        raise ValueError("Invalid paired motion schema: " + str(checks))
    checks["all_finite"] = all(bool(np.isfinite(motion[key]).all()) for key in shapes) and all(bool(np.isfinite(objects[key]).all()) for key in ("trans", "rot"))
    checks["unit_quaternions"] = all(bool(np.allclose(np.linalg.norm(array, axis=1), 1., atol=1e-6, rtol=0.)) for array in (motion["root_rot"], objects["rot"]))
    if "source_time_s" in objects:
        source = objects["source_time_s"]
        checks["capture_clock_shape_finite_monotone"] = source.shape == (frames,) and bool(np.isfinite(source).all()) and bool((np.diff(source) >= 0).all())
    urdf = ET.parse(resmimic_root() / "assets/g1/g1_custom_collision_29dof.urdf").getroot()
    names = [joint.get("name") for joint in urdf.findall("joint") if joint.get("type") != "fixed"]
    limits = joint_limit_report(urdf, motion["dof_pos"], names, fps)
    checks["joint_position_limits"] = all(row["position_violations"] == 0 for row in limits)
    checks["native_joint_velocity_limits"] = all(row["velocity_violations"] == 0 for row in limits)
    forward_kinematics, foot_clearance, clearance_summary = fk_helpers()
    local, rotations = forward_kinematics(urdf, motion["dof_pos"], names, motion["link_body_list"])
    fk_error = float(np.abs(local - motion["local_body_pos"]).max())
    checks["stored_skeleton_matches_urdf_fk"] = fk_error <= 2e-6
    result = dict(status="PASS_PAIRED_SCHEMA_AND_NATIVE_KINEMATICS" if all(checks.values()) else "FAIL", checks=checks,
                  motion=relative(path), frames=frames, fps=fps, duration_endpoint_s=(frames - 1) / fps,
                  maximum_local_fk_error_m=fk_error, joint_limits=limits,
                  feet=clearance_summary(foot_clearance(urdf, local, rotations, motion["link_body_list"], motion["root_pos"], Rotation.from_quat(motion["root_rot"]).as_matrix())),
                  input_sha256={relative(p): sha(p) for p in (path, path.with_suffix(".npz"))},
                  physical_success_verified=False, policy_or_training_run=False,
                  limitations=["Native frames only; no dense-interpolant collision/contact check.", "Finite geometry is not proof of foot support, grasp, force closure, or dynamic feasibility."])
    if report_path:
        dest = output_path(report_path)
        new_files([dest])
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_json(dest, result)
    show(result)
    return 0 if all(checks.values()) else 2


def render_motion(path, mesh_path, output, width=1280, height=720, azimuth=135.):
    np, Rotation, _ = geometry()
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    path, mesh_path = read_path(path), read_path(mesh_path)
    out = output_path(output)
    metadata = out.with_suffix(".json")
    new_files([out, metadata])
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Video dimensions must be positive even numbers")
    motion, objects = load_motion(path)
    count = len(motion["root_pos"])
    if objects["trans"].shape != (count, 3) or objects["rot"].shape != (count, 4):
        raise ValueError("Robot and object frame counts differ")
    sys.path.insert(0, str(resmimic_root() / "legged_gym/legged_gym/scripts"))
    from render_motion import build_scene, set_frame, verify_fk
    model, addresses, root_address, object_id = build_scene(resmimic_root() / "assets/g1/g1_custom_collision_29dof.urdf", mesh_path, width, height)
    data = mujoco.MjData(model)
    verify_fk(model, data, motion, objects, addresses, root_address, object_id)
    world = np.einsum("tij,tkj->tki", Rotation.from_quat(motion["root_rot"]).as_matrix(), motion["local_body_pos"]) + motion["root_pos"][:, None]
    points = np.concatenate((world.reshape(-1, 3), objects["trans"]))
    low, high = points.min(0), points.max(0)
    camera = mujoco.MjvCamera();camera.lookat[:] = (low + high) / 2;camera.distance = max(2.5, float(np.linalg.norm(high - low)) * 1.6)
    camera.azimuth = azimuth;camera.elevation = -15
    options = mujoco.MjvOption();options.geomgroup[0] = 0
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    fps = float(motion["fps"])
    try:
        with imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for index in range(count):
                set_frame(model, data, motion, objects, index, addresses, root_address, object_id)
                renderer.update_scene(data, camera=camera, scene_option=options)
                picture = Image.fromarray(renderer.render());draw = ImageDraw.Draw(picture)
                draw.rectangle((12, 12, min(width - 12, 1080), 88), fill=(245, 248, 250))
                draw.text((24, 20), "REFERENCE REPLAY | stored poses | no policy or dynamics", font=font, fill=(25, 38, 53))
                draw.text((24, 52), "elapsed %.3fs | %.0f Hz | %s" % (index / fps, fps, path.stem), font=font, fill=(50, 65, 80))
                writer.append_data(np.asarray(picture))
    finally:
        renderer.close()
    reader = imageio.get_reader(str(out));video_meta = reader.get_meta_data();decoded = sum(1 for _ in reader);reader.close()
    if decoded != count:
        raise ValueError("Video frame count differs from reference")
    report = dict(status="PASS_REFERENCE_VIDEO", video=relative(out), frames=decoded, fps=video_meta["fps"],
                  dimensions=list(video_meta["size"]), native_duration_endpoint_s=(count - 1) / fps, media_duration_s=count / fps,
                  input_sha256={relative(p): sha(p) for p in (path, path.with_suffix(".npz"), mesh_path)}, video_sha256=sha(out),
                  scope="MuJoCo stored-pose replay; no policy rollout, contact forces, or physical-lifting claim")
    write_json(metadata, report);show(report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check inputs, pinned repositories and geometry dependency availability")
    inspect = sub.add_parser("inspect", help="Inspect trusted raw motion metadata without converting it")
    inspect.add_argument("--dataset", required=True)
    for name in ("align", "retarget"):
        item = sub.add_parser(name, help="Align source clocks" if name == "align" else "Run actual GMR/Mink IK and export a paired reference")
        item.add_argument("--dataset", required=True);item.add_argument("--output", required=True, help="New directory below outputs/; use the same directory for align then retarget")
    validate = sub.add_parser("validate", help="Validate paired schema, native joint limits and target URDF FK")
    validate.add_argument("--motion", required=True);validate.add_argument("--report", help="Optional new JSON below outputs/")
    render = sub.add_parser("render", help="Render a stored reference; this does not run a policy")
    render.add_argument("--motion", required=True);render.add_argument("--object-mesh", required=True);render.add_argument("--output", required=True)
    render.add_argument("--width", type=int, default=1280);render.add_argument("--height", type=int, default=720);render.add_argument("--azimuth", type=float, default=135.)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "doctor":
        return doctor()
    configure_runtime()
    if args.command == "inspect":
        inspect_source(args.dataset)
    elif args.command == "align":
        align(args.dataset, args.output)
    elif args.command == "retarget":
        return retarget(args.dataset, args.output)
    elif args.command == "validate":
        return validate_motion(args.motion, args.report)
    elif args.command == "render":
        render_motion(args.motion, args.object_mesh, args.output, args.width, args.height, args.azimuth)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
