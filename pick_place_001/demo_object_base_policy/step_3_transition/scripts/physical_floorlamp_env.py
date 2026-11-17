"""Local two-actor G1HOI environment for an unchanged blind base actor.

The floorlamp is a free rigid body. Only initial resets may write its pose;
all subsequent motion comes from gravity and collision/contact dynamics.
"""
import json,sys
from pathlib import Path
from object_common import ROOT,REPO
from isaacgym import gymapi,gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from legged_gym.envs.g1.g1_hoi import G1HOI
from legged_gym.envs.g1.g1_motion import G1Motion
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from foot_reference_loader import install_reference_interpolation


def scalar(t):return float(t.detach().cpu())


class PhysicalLampEnv(G1HOI):
    active=False
    def __init__(self,*args,**kwargs):
        self.asset_profile=json.loads((ROOT/'assets/asset_profile.json').read_text())
        cfg=args[0] if args else kwargs['cfg']
        with np.load(cfg.motion.object_motion_file,allow_pickle=False) as z:initial_pos=z['trans'][0].copy();initial_quat=z['rot'][0].copy()
        with np.load(ROOT/'assets/collider_vertices.npz',allow_pickle=False) as z:initial_vertices=z['vertices'].copy()
        gap=float((initial_vertices@Rotation.from_quat(initial_quat).as_matrix().T+initial_pos)[:,2].min())
        correction=max(0.,.002-gap)
        self.asset_profile['initialization'].update(reference_position_m=initial_pos.tolist(),reference_quaternion_xyzw=initial_quat.tolist(),upward_floor_correction_m=correction,source_initial_gap_m=gap)
        self.reset_physics_steps=0;self.object_pose_setter_calls=0;self.object_post_initialization_setter_calls=0
        self.reference_query_calls=0;self.reference_query_clamp_count=0;self.reference_query_max_time_s=0.;self.endpoint_timeout_count=0
        self.last_reset_measurement=None
        super().__init__(*args,**kwargs)
        self._refresh_base_state();self.compute_observations()
        with np.load(ROOT/'assets/collider_vertices.npz',allow_pickle=False) as z:vertices=z['vertices'].copy()
        self.collider_vertices=torch.tensor(vertices,device=self.device,dtype=torch.float32)
        r=Rotation.from_quat(self.asset_profile['initialization']['reference_quaternion_xyzw']).as_matrix()
        self.initial_lamp_up=torch.tensor(r[:,2],device=self.device,dtype=torch.float32)
        assert self.num_actors==2 and self.num_envs==1 and self.obs_buf.shape==(1,1757)

    def create_sim(self):
        if self.headless and self.cfg.env.record_video:self.graphics_device_id=self.sim_device_id
        super().create_sim()

    def _load_object_asset(self):
        options=gymapi.AssetOptions();options.fix_base_link=False;options.disable_gravity=False
        options.default_dof_drive_mode=gymapi.DOF_MODE_NONE;options.vhacd_enabled=False
        options.override_com=False;options.override_inertia=False
        options.linear_damping=.01;options.angular_damping=.01
        options.replace_cylinder_with_capsule=False
        self.target_asset=self.gym.load_asset(self.sim,str(ROOT/'assets'),'floorlamp_physical.urdf',options)
        if self.target_asset is None:raise RuntimeError('Physical floorlamp asset failed to load')
        assert self.gym.get_asset_rigid_body_count(self.target_asset)==1
        assert self.gym.get_asset_dof_count(self.target_asset)==0
        assert self.gym.get_asset_rigid_shape_count(self.target_asset)==3
        # Reference points retain exactly the NPZ/source OBJ coordinate frame.
        import trimesh
        mesh=trimesh.load_mesh(ROOT/'assets/floorlamp_visual.obj',process=False)
        vertices=np.asarray(mesh.vertices);idx=np.linspace(0,len(vertices)-1,1024).astype(int)
        self.object_points=torch.tensor(vertices[idx],device=self.device,dtype=torch.float32)
        self.object_asset_options_record=dict(fix_base_link=False,disable_gravity=False,vhacd_enabled=False,override_com=False,override_inertia=False,linear_damping=.01,angular_damping=.01)

    def _create_envs(self):
        super()._create_envs()
        handle=self.object_handles[0]
        props=self.gym.get_actor_rigid_shape_properties(self.envs[0],handle)
        for p in props:
            p.friction=.6;p.restitution=0.;p.rolling_friction=.001;p.torsion_friction=.01;p.contact_offset=.002;p.rest_offset=0.
        self.gym.set_actor_rigid_shape_properties(self.envs[0],handle,props)
        props=self.gym.get_actor_rigid_body_properties(self.envs[0],handle)
        assert len(props)==1 and abs(props[0].mass-4.)<1e-4
        self.object_body_index=self.gym.find_actor_rigid_body_handle(self.envs[0],handle,'floorlamp')
        assert self.object_body_index==self.num_bodies

    def _load_motions(self):
        install_reference_interpolation();super()._load_motions()
        for name in ['calc_motion_frame','calc_hoi_motion_frame']:
            original=getattr(self._motion_lib,name)
            def clamped(ids,times,method=original):
                lengths=self._motion_lib.get_motion_length(ids);query=torch.minimum(times.clamp_min(0.),(lengths-1e-5).clamp_min(0.))
                self.reference_query_calls+=1;self.reference_query_clamp_count+=int((query!=times).sum())
                if query.numel():self.reference_query_max_time_s=max(self.reference_query_max_time_s,scalar(query.max()))
                return method(ids,query)
            setattr(self._motion_lib,name,clamped)

    def _reset_dofs(self,env_ids,dof_pos,dof_vel):
        ids=env_ids
        self.dof_pos[ids]=self._ref_dof_pos[ids];self.dof_vel[ids]=self._ref_dof_vel[ids]
        actors=self.all_actor_ids[ids,0].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(actors),len(actors))

    def _reset_root_states(self,env_ids,**kwargs):
        ids=env_ids
        if self.active:
            self.object_post_initialization_setter_calls+=1
            raise RuntimeError('Object pose changes are forbidden after physical rollout starts')
        self.episode_init_origin[ids]=self.env_origins[ids]
        self.root_states[ids,:3]=self._ref_root_pos[ids]+self.env_origins[ids]
        self.root_states[ids,3:7]=self._ref_root_rot[ids]
        self.root_states[ids,7:10]=self._ref_root_vel[ids];self.root_states[ids,10:13]=self._ref_root_ang_vel[ids]
        self.object_root_states[ids]=0.
        self.object_root_states[ids,:3]=self._ref_object_root_pos[ids]+self.env_origins[ids]
        self.object_root_states[ids,2]+=self.asset_profile['initialization']['upward_floor_correction_m']
        self.object_root_states[ids,3:7]=self._ref_object_root_rot[ids]
        self.object_root_states[ids,7:10]=torch.tensor(self.cfg.env.object_initial_linear_velocity,device=self.device,dtype=torch.float32)
        self.object_root_states[ids,10:13]=torch.tensor(self.cfg.env.object_initial_angular_velocity,device=self.device,dtype=torch.float32)
        actors=self.all_actor_ids[ids].flatten().to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.all_root_states),gymtorch.unwrap_tensor(actors),len(actors))
        self.object_pose_setter_calls+=1
        self.pre_reset_physics_state=dict(robot_root=self.root_states[0].detach().cpu().tolist(),reference_root_velocity=self._ref_root_vel[0].detach().cpu().tolist(),reference_root_angular_velocity=self._ref_root_ang_vel[0].detach().cpu().tolist(),joint_position_max_error_rad=scalar((self.dof_pos[0]-self._ref_dof_pos[0]).abs().max()),joint_velocity_max_error_rad_s=scalar((self.dof_vel[0]-self._ref_dof_vel[0]).abs().max()),object_root=self.object_root_states[0].detach().cpu().tolist(),object_reference_linear_velocity=self.cfg.env.object_initial_linear_velocity,object_reference_angular_velocity=self.cfg.env.object_initial_angular_velocity)

    def reset_idx(self,ids,motion_ids=None):
        if not len(ids):return
        super().reset_idx(ids,motion_ids=motion_ids)
        self.reset_physics_steps+=1
        self._update_ref_motion();self._refresh_base_state()
        self.last_reset_measurement=dict(robot_root_reference_error_m=scalar(torch.linalg.vector_norm(self.root_states[0,:3]-self._ref_root_pos[0])),
            robot_joint_reference_rms_rad=scalar(torch.sqrt(((self.dof_pos-self._ref_dof_pos)**2).mean())),robot_root_after_initialization=self.root_states[0].detach().cpu().tolist(),robot_joint_velocity_reference_rms_rad_s=scalar(torch.sqrt(((self.dof_vel-self._ref_dof_vel)**2).mean())),object_state_after_initialization=self.object_root_states[0].detach().cpu().tolist())

    def _refresh_base_state(self):G1Motion._refresh_base_state(self)

    def compute_observations(self):
        super().compute_observations()
        rows=self.episode_length_buf<=1
        if bool(rows.any()) and self.cfg.env.history_len>0:
            n=self.cfg.env.n_obs_single;self.obs_buf[rows,n:n*(self.cfg.env.history_len+1)]=self.obs_history_buf[rows].reshape(int(rows.sum()),-1)

    def _get_unified_motion_data(self):
        data=super()._get_unified_motion_data()
        origin=self.episode_init_origin
        data['root_pos']=data['root_pos']+origin[:,None,:]
        bodies=data['whole_key_body_pos_global'];shape=bodies.shape
        data['whole_key_body_pos_global']=(bodies.reshape(shape[0],shape[1],-1,3)+origin[:,None,None,:]).reshape(shape)
        data['root_pos_distance_to_target']=data['root_pos']-self.root_states[:,None,:3]
        return data

    def check_termination(self):
        super().check_termination()
        if not self.active:return
        endpoint=self.episode_length_buf>=self.bound_steps
        self.endpoint_timeout_count+=int(endpoint.sum());self.time_out_buf|=endpoint;self.reset_buf|=endpoint
        if not bool(torch.isfinite(self.object_root_states).all()):self.reset_buf[:]=True
        if bool((torch.linalg.vector_norm(self.object_root_states[:,7:10],dim=-1)>10).any()):self.reset_buf[:]=True

    def object_diagnostics(self):
        state=self.object_root_states[0];rot=Rotation.from_quat(state[3:7].detach().cpu().numpy()).as_matrix()
        vertices=self.collider_vertices.detach().cpu().numpy()@rot.T+state[:3].detach().cpu().numpy()
        gap=float(vertices[:,2].min());tilt=float(np.degrees(np.arccos(np.clip(rot[2,2],-1.,1.))))
        force=self.all_contact_forces[0,self.object_body_index]
        airborne=gap>.02;load=scalar(force[2])>=.5*4.*9.81
        # Only robot and lamp actors plus floor exist. With20mm floor clearance,
        # upward net contact cannot be supplied by the floor's2mm contact margin.
        supported=airborne and load and tilt<20.
        return dict(object_xyz=state[:3].tolist(),reference_object_xyz=self._ref_object_root_pos[0].tolist(),
            object_position_error_m=scalar(torch.linalg.vector_norm(state[:3]-self._ref_object_root_pos[0])),
            object_linear_speed_m_s=scalar(torch.linalg.vector_norm(state[7:10])),object_floor_gap_m=gap,object_tilt_deg=tilt,
            object_contact_force_N=force.tolist(),hand_contact_force_norm_N=torch.linalg.vector_norm(self.contact_forces[0,self.hand_indices],dim=-1).tolist(),
            object_airborne=airborne,robot_supported_airborne_evidence=supported)

    def actual_physical_properties(self):
        p=self.gym.get_actor_rigid_body_properties(self.envs[0],self.object_handles[0])[0]
        inertia=[[p.inertia.x.x,p.inertia.x.y,p.inertia.x.z],[p.inertia.y.x,p.inertia.y.y,p.inertia.y.z],[p.inertia.z.x,p.inertia.z.y,p.inertia.z.z]]
        shapes=self.gym.get_actor_rigid_shape_properties(self.envs[0],self.object_handles[0])
        return dict(object_mass_kg=p.mass,com_m=[p.com.x,p.com.y,p.com.z],inertia_kg_m2=inertia,
            object_body_flags=int(p.flags),object_actor_dof_count=self.gym.get_actor_dof_count(self.envs[0],self.object_handles[0]),
            actor_count=self.gym.get_actor_count(self.envs[0]),actor_indices=self.all_actor_ids.cpu().tolist(),object_body_index=self.object_body_index,
            asset_options=self.object_asset_options_record,object_shapes=[dict(friction=s.friction,restitution=s.restitution,rolling_friction=s.rolling_friction,torsion_friction=s.torsion_friction,contact_offset=s.contact_offset,rest_offset=s.rest_offset) for s in shapes])

    def local_adapter_diagnostics(self):
        return dict(reference_query_calls=self.reference_query_calls,reference_query_clamp_count=self.reference_query_clamp_count,reference_query_max_time_s=self.reference_query_max_time_s,
            pre_reset_physics_state=self.pre_reset_physics_state,reset_physics_steps=self.reset_physics_steps,reset_extra_simulation_time_s=self.reset_physics_steps*self.cfg.sim.dt,last_reset_measurement=self.last_reset_measurement,
            object_pose_setter_calls_during_initialization=self.object_pose_setter_calls,object_pose_setter_calls_after_initialization=self.object_post_initialization_setter_calls,
            endpoint_timeout_count=self.endpoint_timeout_count,external_object_forces_or_constraints=False)

    def render_record(self,mode='rgb_array'):
        self.gym.step_graphics(self.sim)
        for i,camera in enumerate(self._rendering_camera_handles):
            a=self.root_states[i,:3].detach().cpu().numpy();b=self.object_root_states[i,:3].detach().cpu().numpy();target=(a+b)/2;target[2]=.9
            position=target+np.array([0.,-3.3,.75])
            self.gym.set_camera_location(camera,self.envs[i],gymapi.Vec3(*position),gymapi.Vec3(*target))
        self.gym.render_all_camera_sensors(self.sim)
        return [self.gym.get_camera_image(self.sim,self.envs[i],camera,gymapi.IMAGE_COLOR).reshape(960,1440,4) for i,camera in enumerate(self._rendering_camera_handles)]
