"""Scoped static collision oracle copied from the previously reviewed audit.

Only source definitions are retained; the old runtime guard/main are NOT imported.
No mj_step, simulation integration, contact force solve, policy or rendering runs.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from converter_common import REPO, sha
ARM_LINKS = [side+'_'+suffix for side in ['left','right'] for suffix in ['rubber_hand','wrist_yaw_link','elbow_link']]
BODY_LINKS = ['torso_link'] + [side+'_'+suffix for side in ['left','right'] for suffix in ['hip_yaw_link','knee_link','ankle_roll_link']]
LINKS = ARM_LINKS + BODY_LINKS

def numbers(values):
    return ' '.join('%.17g' % float(x) for x in np.asarray(values).ravel())

def urdf_origin(element):
    org = element.find('origin')
    xyz = np.zeros(3) if org is None else np.fromstring(org.get('xyz', '0 0 0'), sep=' ')
    rpy = np.zeros(3) if org is None else np.fromstring(org.get('rpy', '0 0 0'), sep=' ')
    return xyz, Rotation.from_euler('xyz', rpy).as_matrix()

class StaticCollisionOracle:
    """In-memory original mesh convex hulls and exact URDF primitives.

    All bodies are externally positioned mocap bodies. No joint integration,
    contact solver, or physical robot/object asset is created or executed.
    """
    def __init__(self, urdf_path=None, links=None):
        import mujoco
        import trimesh
        self.mujoco = mujoco
        self.urdf_path = Path(urdf_path or REPO / 'assets/g1/g1_custom_collision_29dof.urdf')
        self.links = list(links or LINKS)
        urdf = ET.parse(self.urdf_path).getroot()
        root = ET.Element('mujoco', model='static_contact_shape_oracle')
        ET.SubElement(root, 'compiler', angle='radian')
        ET.SubElement(root, 'option', gravity='0 0 0')
        assets = ET.SubElement(root, 'asset')
        world = ET.SubElement(root, 'worldbody')
        descriptions = {}; mesh_checks = []; source_paths = [self.urdf_path]
        self.geom_names = {}; self.link_collision_descriptions = {}
        for link in self.links:
            source = urdf.find("link[@name='%s']" % link)
            if source is None:
                raise KeyError(link)
            body = ET.SubElement(world, 'body', name=link, mocap='true')
            self.geom_names[link] = {'native': [], 'capsule': []}
            link_descriptions = []
            for index, collision in enumerate(source.findall('collision')):
                position, rotation = urdf_origin(collision)
                quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
                geometry = list(collision.find('geometry'))[0]
                # Collision flags must be enabled at compile time: otherwise
                # MuJoCo omits mesh convex-hull graphs and direct distance
                # queries can return inconsistent distances. We never run
                # collision generation, contact solving, or dynamics here.
                common = dict(pos=numbers(position), quat=numbers(quat), contype='1', conaffinity='1')
                base_name = link + '_collision_' + str(index)
                description = dict(collision_index=index, source_kind=geometry.tag,
                                   source_position_m=position.tolist(), source_rotation_matrix=rotation.tolist())
                if geometry.tag == 'mesh':
                    path = REPO / 'assets/g1' / geometry.get('filename')
                    mesh_name = base_name + '_asset'
                    scale = np.fromstring(geometry.get('scale', '1 1 1'), sep=' ')
                    ET.SubElement(assets, 'mesh', name=mesh_name, file=str(path), scale=numbers(scale))
                    ET.SubElement(body, 'geom', name=base_name, type='mesh', mesh=mesh_name, **common)
                    self.geom_names[link]['native'].append(base_name)
                    self.geom_names[link]['capsule'].append(base_name)
                    source_paths.append(path)
                    source_mesh = trimesh.load_mesh(path, process=True)
                    expected_vertices = transform(np.asarray(source_mesh.vertices) * scale, position, rotation)
                    mesh_checks.append((base_name, mesh_name, expected_vertices))
                    description.update(source=str(path), source_sha256=sha(path),
                                       collision_interpretation='MuJoCo convex mesh distance from source STL vertices; not PhysX cooking proof')
                elif geometry.tag == 'cylinder':
                    radius = float(geometry.get('radius'))
                    half_length = float(geometry.get('length')) / 2
                    for variant, kind in [('native', 'cylinder'), ('capsule', 'capsule')]:
                        name = base_name + '_' + variant
                        ET.SubElement(body, 'geom', name=name, type=kind, size=numbers([radius, half_length]), **common)
                        self.geom_names[link][variant].append(name)
                    description.update(radius_m=radius, native_cylinder_half_length_m=half_length,
                                       capsule_segment_half_length_m=half_length,
                                       collision_interpretation='Native cylinder and capsule surrogate separately; no enclosing box and no added margin')
                elif geometry.tag == 'sphere':
                    ET.SubElement(body, 'geom', name=base_name, type='sphere', size=geometry.get('radius'), **common)
                    self.geom_names[link]['native'].append(base_name)
                    self.geom_names[link]['capsule'].append(base_name)
                elif geometry.tag == 'box':
                    half = np.fromstring(geometry.get('size'), sep=' ') / 2
                    ET.SubElement(body, 'geom', name=base_name, type='box', size=numbers(half), **common)
                    self.geom_names[link]['native'].append(base_name)
                    self.geom_names[link]['capsule'].append(base_name)
                else:
                    raise ValueError('Unsupported primitive ' + geometry.tag)
                link_descriptions.append(description)
            if not self.geom_names[link]['native']:
                raise ValueError('Link has no collision geometry ' + link)
            self.link_collision_descriptions[link] = link_descriptions
        self.xml = ET.tostring(root, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        if hasattr(self.model.opt, 'ccd_tolerance'):
            self.model.opt.ccd_tolerance = 1e-10
        if hasattr(self.model.opt, 'ccd_iterations'):
            self.model.opt.ccd_iterations = 100
        self.data = mujoco.MjData(self.model)
        self.mocap_ids = {link: int(self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)]) for link in self.links}
        self.geom_ids = {link: {variant: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
                                    for variant, names in variants.items()} for link, variants in self.geom_names.items()}
        self.source_hashes = {str(path): sha(path) for path in source_paths}
        mujoco.mj_kinematics(self.model, self.data)
        self.mesh_registration = {}
        for geom_name, mesh_name, expected in mesh_checks:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
            start, count = int(self.model.mesh_vertadr[mesh_id]), int(self.model.mesh_vertnum[mesh_id])
            compiled = self.model.mesh_vert[start:start + count].astype(float)
            world_vertices = transform(compiled, self.data.geom_xpos[geom_id], self.data.geom_xmat[geom_id].reshape(3, 3))
            error = cKDTree(expected).query(world_vertices)[0]
            reverse = cKDTree(world_vertices).query(expected)[0]
            self.mesh_registration[geom_name] = dict(compiled_vertices=count, source_vertices=len(expected),
                    compiled_convex_hull_graph_address=int(self.model.mesh_graphadr[mesh_id]),
                    convex_hull_graph_available=bool(self.model.mesh_graphadr[mesh_id] >= 0),
                    max_compiled_to_source_vertex_error_m=float(error.max()),
                    max_source_to_compiled_vertex_error_m=float(reverse.max()),
                    passed=bool(max(error.max(), reverse.max()) <= 2e-7 and self.model.mesh_graphadr[mesh_id] >= 0),
                    interpretation='Checks automatic MuJoCo mesh centering/rotation against URDF link coordinates')
        self._cache = {}
        self._qp_mesh_equations = {}
        self._previous_separation_normals = {}

    def set_link_poses(self, positions, rotations):
        """Set all link-frame poses and run only mj_kinematics, not dynamics."""
        for i, link in enumerate(self.links):
            mocap = self.mocap_ids[link]
            self.data.mocap_pos[mocap] = positions[i]
            self.data.mocap_quat[mocap] = Rotation.from_matrix(rotations[i]).as_quat()[[3, 0, 1, 2]]
        self.mujoco.mj_kinematics(self.model, self.data)
        self._cache = {}

    def pair_distance(self, link_a, link_b, variant='capsule', max_distance=10.):
        """Minimum signed static proxy distance and witness from A to B."""
        best = None
        all_signed_valid = True
        all_witness_consistent = True
        constituent_pairs = 0
        unresolved_pairs = []
        for a in self.geom_ids[link_a][variant]:
            for b in self.geom_ids[link_b][variant]:
                key = (a, b, float(max_distance))
                if key not in self._cache:
                    witness = np.zeros(6, dtype=np.float64)
                    distance = float(self.mujoco.mj_geomDistance(self.model, self.data, a, b, max_distance, witness))
                    raw_distance, raw_witness = distance, witness.copy()
                    error = abs(float(np.linalg.norm(witness[3:] - witness[:3])) - abs(distance))
                    method = 'MuJoCo_native_CCD'; fallback = None; signed_valid = True; normal_override = None; lower_bound = distance >= max_distance
                    if distance < max_distance:
                        # Some MuJoCo 3.5 CCD queries return zero while their
                        # witness segment is centimetres long. Do not interpret
                        # that numerical failure as contact. Refine separated
                        # convex shapes with an independent convex program.
                        certificate = self._separation_certificate(a, b, raw_witness[3:] - raw_witness[:3])
                        if certificate is not None:
                            distance = certificate['gap_lower_bound_m']
                            witness = np.r_[certificate['point_a_world'], certificate['point_b_world']]
                            normal_override = certificate['normal_a_to_b']
                            method = 'analytic_support_plane_lower_bound' if raw_distance > 1e-9 else 'analytic_support_plane_overrides_CCD_nonpositive'
                            lower_bound = True
                            fallback = certificate
                        elif distance < -1e-9 and error <= 1e-7:
                            # A consistent negative CCD witness with no exact
                            # separating plane remains the signed overlap
                            # measurement; do not replace it with QP zero.
                            pass
                        else:
                            fallback = self._convex_distance_fallback(a, b)
                            if fallback['success'] and fallback.get('support_plane_lower_bound_m', 0.) > 1e-9 and fallback.get('primal_dual_gap_m', np.inf) <= 1e-6:
                                distance = fallback['support_plane_lower_bound_m']
                                witness = np.array(fallback['support_witness_world'])
                                normal_override = np.array(fallback['support_normal_a_to_b'])
                                method = 'convex_QP_with_analytic_support_certificate'; lower_bound = True
                            else:
                                method = 'convex_QP_overlap_or_touch_depth_unresolved' if fallback['success'] else 'MuJoCo_inconsistent_witness_unresolved'
                                signed_valid = False
                    self._cache[key] = (distance, witness.copy(), method, raw_distance, raw_witness, fallback, signed_valid, normal_override, lower_bound)
                distance, witness, method, raw_distance, raw_witness, fallback, signed_valid, normal_override, lower_bound = self._cache[key]
                delta = witness[3:] - witness[:3]
                witness_norm = float(np.linalg.norm(delta))
                witness_error = abs(float(delta @ normal_override) - distance) if normal_override is not None else abs(witness_norm - abs(distance))
                witness_consistent = bool(distance >= max_distance or witness_error <= 1e-7)
                constituent_pairs += 1
                all_signed_valid = all_signed_valid and bool(signed_valid)
                all_witness_consistent = all_witness_consistent and witness_consistent
                if not signed_valid or not witness_consistent:
                    unresolved_pairs.append(dict(geom_a=int(a),geom_b=int(b),method=method,
                                                 distance_m=float(distance),witness_error_m=float(witness_error)))
                if best is None or distance < best['distance_m']:
                    valid = abs(distance) > 1e-9 and np.linalg.norm(delta) > 1e-12 and distance < max_distance
                    normal = np.asarray(normal_override).copy() if normal_override is not None else delta / distance if valid else np.zeros(3)
                    if valid:
                        normal /= np.linalg.norm(normal)
                    best = dict(distance_m=distance, point_a_world=witness[:3].copy(), point_b_world=witness[3:].copy(),
                                normal_a_to_b=normal, normal_valid=bool(valid), geom_a=a, geom_b=b,
                                witness_gap_consistent=witness_consistent, witness_gap_error_m=witness_error,
                                distance_method=method, signed_distance_valid=bool(signed_valid),
                                distance_is_lower_bound=bool(lower_bound),
                                raw_mujoco_distance_m=raw_distance, raw_mujoco_witness_world=raw_witness.copy(),
                                fallback_diagnostics=fallback,
                                maximum_distance_m=max_distance)
        if best is not None:
            best['selected_pair_signed_distance_valid'] = best['signed_distance_valid']
            best['selected_pair_witness_gap_consistent'] = best['witness_gap_consistent']
            best['signed_distance_valid'] = bool(all_signed_valid)
            best['witness_gap_consistent'] = bool(all_witness_consistent)
            best['constituent_pair_count'] = constituent_pairs
            best['unresolved_constituent_pairs'] = unresolved_pairs
        return best

    def _support_point(self, geom, direction):
        """Exact support of compiled source hull or uninflated primitive."""
        m, d, mj = self.model, self.data, self.mujoco
        n = np.asarray(direction, float); n /= np.linalg.norm(n)
        centre = d.geom_xpos[geom]; rotation = d.geom_xmat[geom].reshape(3, 3)
        local_n = rotation.T @ n; kind = int(m.geom_type[geom]); size = m.geom_size[geom]
        if kind == int(mj.mjtGeom.mjGEOM_MESH):
            mid = int(m.geom_dataid[geom]); start, count = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            vertices = m.mesh_vert[start:start + count]
            local = vertices[int(np.argmax(vertices @ local_n))]
        elif kind == int(mj.mjtGeom.mjGEOM_CYLINDER):
            radial = np.linalg.norm(local_n[:2]); local = np.zeros(3)
            if radial > 1e-15:
                local[:2] = size[0] * local_n[:2] / radial
            local[2] = size[1] * np.sign(local_n[2])
        elif kind == int(mj.mjtGeom.mjGEOM_CAPSULE):
            local = size[0] * local_n
            local[2] += size[1] * np.sign(local_n[2])
        elif kind == int(mj.mjtGeom.mjGEOM_SPHERE):
            local = size[0] * local_n
        elif kind == int(mj.mjtGeom.mjGEOM_BOX):
            local = size * np.sign(local_n)
        else:
            raise ValueError('Unsupported support shape')
        return centre + rotation @ local

    def _separation_certificate(self, geom_a, geom_b, witness_direction=None):
        """Positive support-plane gaps are rigorous separation lower bounds."""
        d = self.data
        directions = [d.geom_xpos[geom_b] - d.geom_xpos[geom_a],
                      *np.eye(3), *d.geom_xmat[geom_a].reshape(3, 3).T,
                      *d.geom_xmat[geom_b].reshape(3, 3).T]
        if witness_direction is not None:
            directions.insert(0, witness_direction)
        key = (geom_a, geom_b)
        if key in self._previous_separation_normals:
            directions.insert(0, self._previous_separation_normals[key])
        best = None
        for direction in directions:
            if np.linalg.norm(direction) <= 1e-12:
                continue
            for sign in [1., -1.]:
                normal = sign * np.asarray(direction, float) / np.linalg.norm(direction)
                pa = self._support_point(geom_a, normal)
                pb = self._support_point(geom_b, -normal)
                gap = float((pb - pa) @ normal)
                if gap > 1e-9 and (best is None or gap > best['gap_lower_bound_m']):
                    best = dict(success=True, gap_lower_bound_m=gap, point_a_world=pa,
                                point_b_world=pb, normal_a_to_b=normal,
                                scope='Exact source-shape support plane; lower bound, not necessarily closest-point distance')
                    # A comfortably positive proof is enough; avoid many
                    # redundant support scans in dense free-space audits.
                    if gap >= .005:
                        self._previous_separation_normals[key] = normal
                        return best
        if best is not None:
            self._previous_separation_normals[key] = best['normal_a_to_b']
        return best

    def _convex_distance_fallback(self, geom_a, geom_b):
        """Convex separation QP; overlap depth is deliberately not invented."""
        m, d, mj = self.model, self.data, self.mujoco
        descriptors = []; initial = [0.] * 6; bounds = [(None, None)] * 6
        for side, gid in enumerate([geom_a, geom_b]):
            sl = slice(side * 3, side * 3 + 3)
            centre = d.geom_xpos[gid].copy()
            rotation = d.geom_xmat[gid].reshape(3, 3).copy()
            kind = int(m.geom_type[gid]); size = m.geom_size[gid].copy()
            desc = dict(slice=sl, centre=centre, axis=rotation[:, 2], kind=kind, size=size)
            if kind == int(mj.mjtGeom.mjGEOM_MESH):
                mid = int(m.geom_dataid[gid])
                start, count = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                local = m.mesh_vert[start:start + count].astype(float)
                if mid not in self._qp_mesh_equations:
                    self._qp_mesh_equations[mid] = ConvexHull(local).equations
                equation = self._qp_mesh_equations[mid]
                matrix = equation[:, :3] @ rotation.T
                desc.update(matrix=matrix, offset=-equation[:, 3] + matrix @ centre)
                initial[sl] = transform(local.mean(axis=0)[None], centre, rotation)[0]
            else:
                initial[sl] = centre
                if kind == int(mj.mjtGeom.mjGEOM_CAPSULE):
                    desc['segment_index'] = len(initial)
                    initial.append(0.); bounds.append((-float(size[1]), float(size[1])))
                elif kind not in [int(mj.mjtGeom.mjGEOM_CYLINDER), int(mj.mjtGeom.mjGEOM_SPHERE), int(mj.mjtGeom.mjGEOM_BOX)]:
                    return dict(success=False, reason='Unsupported fallback shape')
                if kind == int(mj.mjtGeom.mjGEOM_BOX):
                    desc['rotation'] = rotation
            descriptors.append(desc)
        initial = np.asarray(initial, float); dimensions = len(initial)
        def constraints(x, derivative=False):
            values = []; rows = []
            for desc in descriptors:
                sl = desc['slice']; p = x[sl]; delta = p - desc['centre']; kind = desc['kind']; axis = desc['axis']
                if 'matrix' in desc:
                    a = desc['matrix']; v = desc['offset'] - a @ p
                    j = np.zeros((len(v), dimensions)); j[:, sl] = -a
                elif kind == int(mj.mjtGeom.mjGEOM_BOX):
                    a = np.r_[desc['rotation'].T, -desc['rotation'].T]
                    v = np.r_[desc['size'], desc['size']] - a @ delta
                    j = np.zeros((6, dimensions)); j[:, sl] = -a
                else:
                    if kind == int(mj.mjtGeom.mjGEOM_CYLINDER):
                        axial = float(delta @ axis); radial = delta - axial * axis
                        v = np.array([desc['size'][0] ** 2 - radial @ radial, desc['size'][1] - axial, desc['size'][1] + axial])
                        j = np.zeros((3, dimensions)); j[0, sl] = -2 * radial; j[1, sl] = -axis; j[2, sl] = axis
                    else:
                        vector = delta.copy()
                        if 'segment_index' in desc:
                            vector -= axis * x[desc['segment_index']]
                        v = np.array([desc['size'][0] ** 2 - vector @ vector])
                        j = np.zeros((1, dimensions)); j[0, sl] = -2 * vector
                        if 'segment_index' in desc:
                            j[0, desc['segment_index']] = 2 * float(vector @ axis)
                values.extend(v); rows.extend(j)
            return np.asarray(rows) if derivative else np.asarray(values)
        def objective(x):
            return .5 * float(np.sum((x[:3] - x[3:6]) ** 2))
        def gradient(x):
            result = np.zeros(dimensions); delta = x[:3] - x[3:6]
            result[:3] = delta; result[3:6] = -delta
            return result
        result = minimize(objective, initial, jac=gradient, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints, 'jac': lambda x: constraints(x, True)},
                          options={'ftol': 1e-14, 'maxiter': 100})
        violation = float(max(0., -constraints(result.x).min()))
        distance = float(np.linalg.norm(result.x[:3] - result.x[3:6]))
        support = None
        if distance > 1e-9:
            normal = (result.x[3:6] - result.x[:3]) / distance
            pa = self._support_point(geom_a, normal); pb = self._support_point(geom_b, -normal)
            gap = float((pb - pa) @ normal)
            support = dict(support_plane_lower_bound_m=gap, primal_dual_gap_m=distance - gap,
                           support_witness_world=np.r_[pa, pb].tolist(), support_normal_a_to_b=normal.tolist())
        certificate_valid = bool(support is not None and support['support_plane_lower_bound_m'] > 1e-9
                                 and abs(support['primal_dual_gap_m']) <= 1e-7 and violation <= 1e-8)
        # A primal/dual geometric certificate is independent of SLSQP's
        # line-search termination flag; preserve that flag separately.
        report = dict(success=bool(certificate_valid or (result.success and violation <= 1e-8)),
                    optimizer_terminated_successfully=bool(result.success), separation_certificate_valid=certificate_valid,
                    distance_m=float(np.linalg.norm(result.x[:3] - result.x[3:6])),
                    witness_world=result.x[:6].tolist(), maximum_primal_constraint_violation=violation,
                    iterations=int(result.nit), message=str(result.message),
                    scope='Independent convex closest-point optimization; a zero result cannot recover signed penetration depth')
        if support is not None:
            report.update(support)
        return report

def transform(vertices, position, rotation):
    return np.asarray(vertices) @ np.asarray(rotation).T + np.asarray(position)
